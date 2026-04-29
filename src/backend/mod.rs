pub mod dummy;
pub mod traits;

use traits::Backend;

pub fn create_backend(backend_type: &str) -> Box<dyn Backend> {
    match backend_type {
        "dummy" => Box::new(dummy::DummyBackend::new(32000, 4096, 32)),
        _ => {
            tracing::warn!("Unknown backend '{backend_type}', falling back to dummy");
            Box::new(dummy::DummyBackend::new(32000, 4096, 32))
        }
    }
}
