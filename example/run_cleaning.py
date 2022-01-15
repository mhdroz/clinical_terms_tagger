import os, sys
import time
import glob
import re

#python setup_cleaning.py start_idx, end_idx, threads

job_start = int(sys.argv[1])
job_end = int(sys.argv[2])
threads = int(sys.argv[3])
joblist = list(range(job_start, job_end, 1))

for jobid in joblist:

    os.chdir('batch%03d/src' % jobid)

    header_file = 'HP.py'
    
    os.system('sed -i "s/BATCHID/%03d/g" %s' % (jobid, header_file))
    os.system('sed -i "s/THREADS/%d/g" %s' % (threads, header_file))
    print('starting the job:')
    os.system("nohup python cleanup_extractions.py > cleanup.out")
    time.sleep(0.5)

    os.chdir('../../')
