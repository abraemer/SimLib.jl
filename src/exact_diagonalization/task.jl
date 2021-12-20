abstract type EDTask end

# output arrays
function initialize! end

# can preallocate compute space
# Note: This needs to wrap the Task in an additional struct
# The additional memory may not be a part of the same struct
# otherwise the code breaks in a multithreaded (not multiprocessed) setting
# as all threads share the same memory location for their temporary results
initialize_local(task::EDTask) = task

# task, ρindex, shot, fieldindex, eigen
function compute_task! end

# task, ρindex, shot, fieldindex
function failed_task! end

# task, EDDataDescriptor -> Data object
function assemble end
