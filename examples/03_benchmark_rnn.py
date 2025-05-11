from tsfast.basics import *
import identibench as idb
#build_model function using the IdentiBench API
def build_model(context: idb.TrainingContext):
    #dataloader from benchmark speciication
    dls = create_dls_from_spec(context.spec)
    #RNN Learner with provided configuration
    lrn = RNNLearner(dls, 
                     rnn_type=context.hyperparameters['model_type'], 
                     num_layers=context.hyperparameters['num_layers'], 
                     n_skip=context.spec.init_window)
    #training with 
    lrn.fit_flat_cos(n_epoch=1)
    model = InferenceWrapper(lrn)
    return model
# configure model and benchmarks
model_config = {'model_type':'lstm','num_layers':2}
benchmarks = idb.workshop_benchmarks.values()
# test identification method on all provided benchmark datasets
results = [idb.run_benchmark(spec,build_model,model_config)
            for spec in benchmarks]