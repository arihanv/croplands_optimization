import sys
# sys.path.append('../pySOT/')
# from src import *
import logging
from poap.controller import ThreadController, BasicWorkerThread
import numpy as np
import os.path
from pySOT_torch import TorchOptim
from datetime import datetime
from pySOT.experimental_design import *
# from pySOT import *
from pySOT.surrogate import *
from pySOT.strategy import *
import time

def main():
    log_file = os.path.splitext(__file__)[0]+".log"
    millis = int(round(time.time() * 1000))
    print('Started: ' + str(datetime.now()) + ' (' + str(millis) + ')')
    if not os.path.exists("./logfiles"):
        os.makedirs("logfiles")
    if os.path.exists("./logfiles/"+log_file):
        os.remove("./logfiles/"+log_file)
    logging.basicConfig(filename="./logfiles/"+log_file,
                        level=logging.INFO)

    nthreads = int(sys.argv[1])
    maxeval = 50
    seed = sys.argv[3]
    server = sys.argv[4]

    np.random.seed(int(seed))

    print("\nNumber of threads: "+str(nthreads))
    print("Maximum number of evaluations: "+str(maxeval))
    print("Search strategy: Candidate DyCORS")
    print("Experimental design: Latin Hypercube")
    print("Surrogate: Cubic RBF")
    print('best\tf_eval_time\tresult\ttestset_result\tf_eval_count\twallclock_time\thyper-parameters')
    nsamples = nthreads

    data = TorchOptim(seed=seed, server=server)

    
    # Create a strategy and a controller
    controller = ThreadController()
    controller.strategy = \
        SyncStrategyNoConstraints(
            worker_id=0, data=data,
            maxeval=maxeval, nsamples=nsamples,
            exp_design=LatinHypercube(dim=data.dim, npts=2*(data.dim+1)),
            response_surface=RBFInterpolant(surftype=CubicRBFSurface, maxp=maxeval),
            sampling_method=CandidateDYCORS(data=data, numcand=100*data.dim))

    # Launch the threads and give them access to the objective function
    for _ in range(nthreads):
        worker = BasicWorkerThread(controller, data.objfunction)
        controller.launch_worker(worker)

    # Run the optimization strategy
    result = controller.run()

    print('Best value found: {0}'.format(result.value))
    print('Best solution found: {0}\n'.format(
        np.array_str(result.params[0], max_line_width=np.inf,
                     precision=5, suppress_small=True)))

    millis = int(round(time.time() * 1000))
    print('Ended: ' + str(datetime.now()) + ' (' + str(millis) + ')')

if __name__ == '__main__':
    main()