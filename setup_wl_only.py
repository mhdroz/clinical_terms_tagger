import os, sys
import time
import glob
import re

job_start = int(sys.argv[1])
job_end = int(sys.argv[2])
joblist = list(range(job_start, job_end, 1))

for jobid in joblist:
        
    os.system('mkdir batch%03d' % jobid)
    os.system('cp -r src batch%03d' % jobid)
    os.chdir('batch%03d/src' % jobid)

    header_file = 'HP.py'
    
    os.system('sed -i "s/BATCHID/%03d/g" %s' % (jobid, header_file))
    print('starting the job:')
    os.system("nohup python meddra_extract.py --task 'label' --label_model 'label_model'  > label_meddra.out &")
    time.sleep(0.5)

    os.chdir('../../')
