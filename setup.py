import os, sys
import time
import glob
import re

job_start = int(sys.argv[1])
job_end = int(sys.argv[2])
joblist = list(range(job_start, job_end, 1))

for jobid in joblist:

    #print(jobid)
    #if ('batch%02d' %jobid):
    #    print('batch%02d already exisits, starting the calculation:' % jobid)
    #else:
        
    os.system('mkdir batch%03d' % jobid)
    os.system('cp -r src batch%03d' % jobid)
    #os.system('cp *sh batch%02d' % jobid)
    os.chdir('batch%03d/src' % jobid)

    #slurm_file = 'run.sh'
    #slurm_gpu = 'run-gpu.sh'
    header_file = 'HP.py'
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_file))
    #os.system('sed -i "s/BATCHID/%02d/g" %s' % (jobid, slurm_gpu))
    #os.system('sed -i "s/BATCHID2/%02d/g" %s' % (jobid, input_file))
    os.system('sed -i "s/BATCHID/%03d/g" %s' % (jobid, header_file))
    print('starting the job:')
    os.system("nohup python meddra_extract.py --label_model 'label_model' --task 'multi' --threads 28 --datasource bucket > extract_meddra.out && \
            python meddra_extract.py --label_model 'label_model' --task 'label' > label_meddra.out")
    time.sleep(0.5)

    os.chdir('../../')
