use rdkafka::config::ClientConfig;
use rdkafka::producer::BaseProducer;

fn main() {
    // rd_kafka_new pulls rdkafka_ssl.o (rd_kafka_ssl_ctx_init and its ENGINE_* calls) into the link.
    let producer: Result<BaseProducer, _> = ClientConfig::new()
        .set("bootstrap.servers", "127.0.0.1:1")
        .set("security.protocol", "ssl")
        .create();
    println!("producer created: {}", producer.is_ok());
}
