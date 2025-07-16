MODEL_DICT = {}
DATASET_DICT = {}
COLLATOR_DICT = {}
MODEL_ARG_FUNC_DICT = {}
EXPLAIN_DICT = {}

def register_model(name):
    def decorator(model_cls):
        MODEL_DICT[name] = model_cls
        return model_cls
    return decorator 

def register_dataset(name):
    def decorator(dataset_cls):
        DATASET_DICT[name] = dataset_cls
        return dataset_cls
    return decorator 

def register_collator(name):
    def decorator(collator_cls):
        COLLATOR_DICT[name] = collator_cls
        return collator_cls
    return decorator

def register_model_arg_func(name):
    def decorator(func):
        MODEL_ARG_FUNC_DICT[name] = func
        return func
    return decorator

def register_explain(name):
    def decorator(func):
        EXPLAIN_DICT[name] = func
        return func
    return decorator