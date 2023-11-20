import time
import cProfile


def time_this(func):
    """
    Decorator function to measure the execution time of a given function.

    Parameters:
        func (callable): The function to be measured.

    Returns:
        callable: A wrapped function that returns the original function's result
                  along with the execution time in seconds.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper


def profile_this(func):
    """
    Decorator function to profile the execution of a given function using cProfile.

    Parameters:
        func (callable): The function to be profiled.

    Returns:
        callable: A wrapped function that returns the original function's result
                  and prints a cumulative profile of function calls.
    """

    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort="cumulative")
        return result

    return wrapper
