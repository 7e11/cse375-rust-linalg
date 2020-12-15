use rustacuda::prelude::*;
use rustacuda::launch;
use std::error::Error;
use std::ffi::CString;

// Comparison linalg libraries
use ndarray::{Array, ShapeBuilder};
use std::time::Instant;
use linalg::cuda_dot;


/// Examples:
/// Adding two numbers: https://bheisler.github.io/RustaCUDA/rustacuda/index.html
/// Adding two arrays of numbers: https://bheisler.github.io/RustaCUDA/rustacuda/macro.launch.html
///
fn main() {
    benchmark_ndarray_multiplication();
    rust_cuda();
    test_dot_impl();
}

fn test_dot_impl() {
    let shape = 10;
    let a = Array::from_shape_simple_fn((shape, shape).set_f(false), || 1f32);
    let b = Array::from_shape_simple_fn((shape, shape).set_f(false), || 2f32);
    let start = Instant::now();
    let c = cuda_dot(a.view(), b.view());
    println!("{:?}", c);
    println!("Elapsed Time: {:?}", start.elapsed());
}

fn benchmark_ndarray_multiplication() {
    let shape = 10;
    let a = Array::from_shape_simple_fn((shape, shape), || 1f32);
    let b = Array::from_shape_simple_fn((shape, shape), || 2f32);
    let start = Instant::now();
    let c = a.dot(&b);
    println!("{:?}", c);
    println!("Elapsed Time: {:?}", start.elapsed());
}

const SHAPE: usize = 10;
fn rust_cuda() -> Result<(), Box<dyn Error>> {
    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty())?;

    // Get the first device
    let device = Device::get_device(0)?;

    // Create a context associated to this device
    let _context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("../resources/add.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;


    let mut in_x = DeviceBuffer::from_slice(&[1.0f32; SHAPE*SHAPE])?;
    let mut in_y = DeviceBuffer::from_slice(&[2.0f32; SHAPE*SHAPE])?;
    let mut out = DeviceBuffer::from_slice(&[0.0f32; SHAPE*SHAPE])?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    let start = Instant::now();
    unsafe {
        let result = launch!(module.mm_kernel<<<(64, 64, 1), (16, 16, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out.as_device_ptr(),
            SHAPE
        ));
        result?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;
    println!("Elapsed Time after synchronize: {:?}", start.elapsed());

    // Copy the results back to host memory
    let mut out_host = [0.0f32; SHAPE*SHAPE];
    out.copy_to(&mut out_host)?;
    println!("Elapsed Time after copy back to host mem: {:?}", start.elapsed());

    println!("{:?}", out_host);
    Ok(())
}