use rustacuda::prelude::*;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use std::error::Error;
use std::ffi::CString;

/// Examples:
/// Adding two numbers: https://bheisler.github.io/RustaCUDA/rustacuda/index.html
/// Adding two arrays of numbers: https://bheisler.github.io/RustaCUDA/rustacuda/macro.launch.html
///
fn main() -> Result<(), Box<dyn Error>> {
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

    let mut in_x = DeviceBuffer::from_slice(&[1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    let mut in_y = DeviceBuffer::from_slice(&[5.0f32, 6.0f32, 7.0f32, 8.0f32])?;
    let mut out = DeviceBuffer::from_slice(&[0.0f32; 4])?;

    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        let function_name = CString::new("mm_kernel")?;
        let mm_kernel = module.get_function(&function_name)?;
        // Launch with 1x1x1 (1) blocks of 10x1x1 (10) threads, to show that you can use tuples to
        // configure grid and block size.
        let result = launch!(mm_kernel<<<1, 3, 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out.as_device_ptr(),
            out.len()
        ));
        result?;
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize()?;

    // Copy the results back to host memory
    let mut out_host = [0.0f32; 4];
    out.copy_to(&mut out_host)?;

    println!("{:?}", out_host);
    Ok(())
}