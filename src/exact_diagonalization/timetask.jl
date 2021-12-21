mutable struct TimeTask{T} <: EDTask
    elapsed::UInt64
    allocated::Int
    gctime::Int
    task::T
    TimeTask(task) = new{typeof(task)}(0,0,0,task)
end


initialize!(tt::TimeTask, arrayconstructor, spectral_size) = initialize!(tt.task, arrayconstructor, spectral_size)
# TODO: not thread/multiprocess safe!!!
# atomics? for threaded case
# processes -> allocate one counter for each worker
initialize_local(tt::TimeTask) = (tt.task = initialize_local(tt.task); tt)
function compute_task!(tt::TimeTask, args...)
    gcbefore = Base.gc_num()
    timebefore = Base.time_ns()

    compute_task!(tt.task, args...)

    timeafter = Base.time_ns()
    gcdiff = Base.GC_Diff(Base.gc_num(), gcbefore)

    tt.elapsed += timeafter - timebefore
    tt.allocated += gcdiff.allocd
    tt.gctime += gcdiff.total_time
end
failed_task!(tt::TimeTask, args...) = failed_task!(tt.task, args...)

function assemble(tt::TimeTask, edd)
    println("Task time summary:\nElapsed: ", tt.elapsed/10e8, "s\nAllocated: ", tt.allocated, " bytes\nGC time: ", tt.gctime/10e8, "s\nTask: ", typeof(tt.task), "\n")
    assemble(tt.task, edd)
end
