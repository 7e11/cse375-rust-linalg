use rustacuda::prelude::*;
use rustacuda::launch;
use std::ffi::CString;
use ndarray::{OwnedRepr, ArrayBase, Array2, ArrayView2, Array, ShapeBuilder};


// fn dot(a: &ArrayBase<OwnedRepr<f32>, Ix2>, b: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Array2<A> {
//     let a = a.view();
//     let b = b.view();
//     let ((m, k), (k2, n)) = (a.dim(), b.dim());
//     if k != k2 || m.checked_mul(n).is_none() {
//         // dot_shape_error(m, k, k2, n);
//     }
//
//     let lhs_s0 = a.strides()[0];
//     let rhs_s0 = b.strides()[0];
//     let column_major = lhs_s0 == 1 && rhs_s0 == 1;
//     // If we meet certain criteria, we can do the CUDA implementation
//     return if !column_major && (m > 32 || k > 32 || n > 32) && m == k && k == n && same_type::<A, f32>() {
//         cuda_dot(a, b)
//     } else {
//     // A is Copy so this is safe
//     let mut v = Vec::with_capacity(m * n);
//     let mut c;
//     unsafe {
//         v.set_len(m * n);
//         c = Array::from_shape_vec_unchecked((m, n).set_f(column_major), v);
//     }
//     mat_mul_impl(A::one(), &a, &b, A::zero(), &mut c.view_mut());
//     c
//     }
// }

pub fn cuda_dot<A>(a: ArrayView2<'_, A>, b: ArrayView2<'_, A>) -> Array2<A> {
    // FIXME: Support nonsquare matricies
    let (square_len, _) = a.dim();
    let mut out_host = Vec::with_capacity(square_len * square_len);
    unsafe {
        cuda_dot_impl(a.as_ptr() as *const _,
                      b.as_ptr() as *const _,
                      out_host.as_mut_ptr() as *mut _,
                      square_len);
    }
    // Pinky promise to rust that we've changed it to the necessary length
    unsafe { out_host.set_len(square_len * square_len); }
    let out_array = unsafe { Array::from_shape_vec_unchecked((square_len, square_len).set_f(false), out_host) };
    out_array
}

unsafe fn cuda_dot_impl(a: *const f32, b: *const f32, c: *mut f32, square_len: usize) {
    let len = square_len * square_len;

    // Initialize the CUDA API
    rustacuda::init(CudaFlags::empty()).unwrap();

    // Get the first device
    let device = Device::get_device(0).unwrap();

    // Create a context associated to this device
    let _context = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();

    // Load the module containing the function we want to call
    let module_data = CString::new(include_str!("../resources/add.ptx")).unwrap();
    let module = Module::load_from_string(&module_data).unwrap();

    // Create a stream to submit work to
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let out = vec![0.0f32; len];
    let mut out = DeviceBuffer::from_slice(&out).unwrap();

    // Creating stuff on the device can be unsafe. We need to get a slice from a pointer, which is
    // Unsafe behaviour
    // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
    // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
    unsafe {
        let mut in_x = DeviceBuffer::from_slice(std::slice::from_raw_parts(a, len)).unwrap();
        let mut in_y = DeviceBuffer::from_slice(std::slice::from_raw_parts(b, len)).unwrap();

        let result = launch!(module.mm_kernel<<<(64, 64, 1), (16, 16, 1), 0, stream>>>(
            in_x.as_device_ptr(),
            in_y.as_device_ptr(),
            out.as_device_ptr(),
            square_len
        ));
        result.unwrap();
    }

    // The kernel launch is asynchronous, so we wait for the kernel to finish executing
    stream.synchronize().unwrap();

    // Copy the results back to host memory
    let mut out_host = vec![0.0f32; len];
    out.copy_to(&mut out_host);

    // Now copy that into c
    std::ptr::copy(out_host.as_ptr(), c, len);
}
