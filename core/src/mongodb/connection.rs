use async_stream::stream;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::sql::TableReference;
use futures::TryStreamExt;
use mongodb::{
    bson::{doc, Bson, Document},
    options::Collation,
    Client, Collection,
};
use snafu::prelude::*;
use std::sync::Arc;

use crate::mongodb::projection::project_bson_document;
use crate::mongodb::utils::arrow::mongo_docs_to_arrow;
use crate::mongodb::utils::schema::infer_arrow_schema_from_documents;
use crate::mongodb::utils::unnest::{unnest_bson_documents, UnnestBehavior, UnnestParameters};
use crate::mongodb::{Error, QuerySnafu, Result, UnableToGetSchemaSnafu, UnableToGetTablesSnafu};
use crate::schema_projection::SchemaProjection;
use crate::util::schema::merge_inferred_and_declared_schemas;

/// How a `find` on a collection compares strings, which decides the string
/// predicates it can evaluate as SQL does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringComparison {
    /// By code point, as SQL does: the collection has no collation, or the
    /// simple one.
    CodePoint,
    /// By the collection's default collation, which a `find` replaces with
    /// the simple one when it has to compare strings by code point.
    Collated,
    /// By a collation a `find` cannot replace, such as a view's, or by one
    /// that could not be read.
    Fixed,
}

impl StringComparison {
    /// The collation a `find` with `filter` has to name to compare strings as
    /// SQL does, if any.
    #[must_use]
    pub fn collation_for(self, filter: &Document) -> Option<Collation> {
        (self == Self::Collated && crate::mongodb::utils::expression::orders_strings(filter))
            .then(|| Collation::builder().locale("simple").build())
    }
}

pub struct MongoDBConnection {
    pub client: Arc<Client>,
    pub db_name: String,
    tz: Option<String>,
    unnest_parameters: UnnestParameters,
    num_documents_to_infer_schema: i64,
}

impl MongoDBConnection {
    pub fn new(
        client: Arc<Client>,
        db_name: String,
        tz: Option<String>,
        unnest_parameters: UnnestParameters,
        num_documents_to_infer_schema: i64,
    ) -> Self {
        MongoDBConnection {
            client,
            db_name,
            tz,
            unnest_parameters,
            num_documents_to_infer_schema,
        }
    }

    fn get_collection(&self, collection: &str) -> Collection<Document> {
        self.client.database(&self.db_name).collection(collection)
    }

    /// How a `find` on `collection` compares strings, from the collation the
    /// collection or view was created with.
    pub async fn string_comparison(&self, collection: &str) -> StringComparison {
        let listing = self
            .client
            .database(&self.db_name)
            .run_command(doc! { "listCollections": 1, "filter": { "name": collection } })
            .await;
        let specification = match &listing {
            Ok(listing) => listing
                .get_document("cursor")
                .and_then(|cursor| cursor.get_array("firstBatch"))
                .ok()
                .and_then(|batch| batch.first())
                .and_then(Bson::as_document),
            Err(_) => None,
        };
        let Some(specification) = specification else {
            tracing::debug!(
                "Could not read the collation of MongoDB collection '{collection}', so its string inequalities and ranges are not pushed down to MongoDB"
            );
            return StringComparison::Fixed;
        };
        let collation = specification
            .get_document("options")
            .ok()
            .and_then(|options| options.get("collation"));
        let code_point = match collation {
            None => true,
            Some(Bson::Document(collation)) => collation
                .get_str("locale")
                .is_ok_and(|locale| locale == "simple"),
            Some(_) => false,
        };
        match specification.get_str("type") {
            _ if code_point => StringComparison::CodePoint,
            // A view, or a time series collection, refuses a collation other
            // than its own.
            Ok("collection") => StringComparison::Collated,
            _ => StringComparison::Fixed,
        }
    }

    pub async fn tables(&self) -> Result<Vec<String>, Error> {
        let db = self.client.database(&self.db_name);
        let collections = db
            .list_collection_names()
            .await
            .boxed()
            .context(UnableToGetTablesSnafu)?;
        Ok(collections)
    }

    /// Get the schema for a collection, optionally merging with a declared schema.
    ///
    /// * If the collection has documents, the inferred schema is merged with
    ///   the declared schema (declared fields take precedence over inferred ones
    ///   with the same name; extra declared fields are appended).
    /// * If the collection is empty and a declared schema is provided, the
    ///   declared schema is returned directly — no error, no retry needed.
    /// * If the collection is empty and no declared schema is provided, the
    ///   `EmptyCollection` error is returned.
    pub async fn get_schema(
        &self,
        table_reference: &TableReference,
        declared_schema: Option<SchemaRef>,
    ) -> Result<SchemaRef, Error> {
        let collection_name = table_reference.table();
        let coll = self.get_collection(collection_name);

        let sample = coll
            .find(doc! {})
            .limit(self.num_documents_to_infer_schema)
            .await
            .boxed()
            .context(UnableToGetSchemaSnafu)?;

        let docs: Vec<Document> = sample
            .try_collect()
            .await
            .boxed()
            .context(UnableToGetSchemaSnafu)?;

        let unnested_docs = match self.unnest_parameters.behavior {
            UnnestBehavior::Depth(0) => docs,
            _ => unnest_bson_documents(docs, &self.unnest_parameters)?,
        };

        if unnested_docs.is_empty() {
            return match declared_schema {
                Some(schema) => Ok(schema),
                None => Err(Error::EmptyCollection {
                    collection_name: collection_name.to_string(),
                }),
            };
        }

        let inferred = infer_arrow_schema_from_documents(
            collection_name,
            &unnested_docs,
            self.tz.clone().as_deref(),
        )
        .boxed()
        .context(UnableToGetSchemaSnafu)?;

        Ok(merge_inferred_and_declared_schemas(
            inferred,
            declared_schema.as_ref(),
        ))
    }

    pub async fn query_arrow(
        &self,
        table_reference: &Arc<TableReference>,
        projected_schema: &SchemaRef,
        filters_doc: &Document,
        collation: Option<&Collation>,
        limit: Option<i32>,
        schema_projection: Option<&SchemaProjection>,
    ) -> Result<SendableRecordBatchStream> {
        let collection_name = table_reference.table();
        let coll = self.get_collection(collection_name);

        // With a JSON-nesting catch-all the projected schema's `data` column is
        // not a real MongoDB field; we must fetch the full documents so every
        // non-declared field can be folded into the catch-all.
        let nesting = schema_projection.filter(|p| p.has_catch_all());
        let mongo_projection = if nesting.is_some() {
            Document::new()
        } else {
            schema_to_mongo_projection(projected_schema)
        };

        let mut find = coll.find(filters_doc.clone()).projection(mongo_projection);

        if let Some(collation) = collation {
            find = find.collation(collation.clone());
        }

        if let Some(l) = limit {
            find = find.limit(l.into());
        }

        let cursor = find.await.boxed().context(QuerySnafu)?;
        let chunked_stream = cursor.try_chunks(4_000);
        let projected_schema_clone = Arc::clone(projected_schema);
        let unnest_parameters = self.unnest_parameters.clone();
        let projection = schema_projection.filter(|p| p.has_catch_all()).cloned();

        let schema = Arc::clone(projected_schema);
        let batch_stream = stream! {
            for await chunk in chunked_stream {
                match chunk {
                    Ok(docs) => {
                        let unnested_docs = match unnest_parameters.behavior {
                            UnnestBehavior::Depth(0) => docs,
                            _ => {
                                unnest_bson_documents(docs, &unnest_parameters)?
                            }
                        };

                        // Fold non-declared fields into the catch-all before
                        // Arrow conversion when JSON nesting is configured.
                        let projected_docs = match &projection {
                            Some(p) => unnested_docs
                                .into_iter()
                                .map(|d| project_bson_document(d, p))
                                .collect(),
                            None => unnested_docs,
                        };

                        let batch = mongo_docs_to_arrow(&projected_docs, Arc::clone(&projected_schema_clone))?;
                        yield Ok(batch);
                    }
                    Err(e) => yield Err(Error::QueryError { source: Box::new(e) }),
                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            Box::pin(batch_stream)
                .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string())),
        )))
    }
}

pub fn schema_to_mongo_projection(projected_schema: &SchemaRef) -> Document {
    let mut projection = Document::new();

    if projected_schema.fields().is_empty() {
        return projection;
    }

    let has_id = projected_schema.fields().iter().any(|f| f.name() == "_id");

    let names: Vec<&str> = projected_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    for name in &names {
        // MongoDB refuses a projection of a path beside one of its prefixes as
        // a path collision. The prefix returns the whole embedded document,
        // which unnesting flattens into the path's column too.
        let covered = name
            .match_indices('.')
            .any(|(i, _)| names.contains(&&name[..i]));
        if !covered {
            projection.insert(*name, 1);
        }
    }

    // MongoDB always includes _id unless explicitly suppressed
    if !has_id {
        projection.insert("_id", 0);
    }

    projection
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use mongodb::bson::doc;

    #[test]
    fn a_path_is_not_projected_beside_its_prefix() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Utf8, false),
            Field::new("address", DataType::Utf8, true),
            Field::new("address.city", DataType::Utf8, true),
            Field::new("address.zip.code", DataType::Utf8, true),
            Field::new("addressbook", DataType::Utf8, true),
        ]));
        assert_eq!(
            schema_to_mongo_projection(&schema),
            doc! { "_id": 1, "address": 1, "addressbook": 1 }
        );
    }

    #[test]
    fn test_projection_with_id() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let projection = schema_to_mongo_projection(&schema);
        assert_eq!(projection, doc! { "_id": 1, "name": 1 });
    }

    #[test]
    fn test_projection_without_id_suppresses() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]));
        let projection = schema_to_mongo_projection(&schema);
        assert_eq!(projection, doc! { "name": 1, "age": 1, "_id": 0 });
    }

    #[test]
    fn test_projection_empty_schema() {
        let schema = Arc::new(Schema::empty());
        let projection = schema_to_mongo_projection(&schema);
        assert!(projection.is_empty());
    }

    #[test]
    fn test_projection_single_field() {
        let schema = Arc::new(Schema::new(vec![Field::new("email", DataType::Utf8, true)]));
        let projection = schema_to_mongo_projection(&schema);
        assert_eq!(projection, doc! { "email": 1, "_id": 0 });
    }
}
