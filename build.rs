fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &[
                "proto/inference.proto",
                "proto/control.proto",
            ],
            &["proto/"],
        )?;
    Ok(())
}
