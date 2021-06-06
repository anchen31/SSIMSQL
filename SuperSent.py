import os                                                                       
from multiprocessing import Pool                                                
                                                                                
                                                                                
processes = ('TSSmySQL.py', 'RSSmySQL.py', 'ibStream.py')                                    
other = ('stonks.py',)
                                                  
                                                                                
def run_process(process):                                                             
    os.system('python {}'.format(process))                                       
                                                                                
                                                                                
pool = Pool(processes=3)                                                        
pool.map(run_process, processes) 
pool.map(run_process, other) 