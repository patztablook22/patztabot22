# keeping this in a separate file due to python module caching system
#
# essentially, i can reload the other source code files while
# keeping this one loaded all the time, avoiding memory leaks etc.

llm_cache = {}
