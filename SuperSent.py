# import os
# from multiprocessing import Pool


# def run_process(process):                                                             
#     os.system('python {}'.format(process))                                                           


# #pool.map(run_process, other)                                                 
# def main():

#     processes = ('TSSmySQL.py', 'RSSmySQL.py')
#     #processes = ('TSSmySQL.py', 'RSSmySQL.py', 'ibStream.py')                                   
#     #other = ('stonks.py',)
#     pool = Pool(processes=2)                                                        
#     pool.map(run_process, processes)

# if __name__ == '__main__':
#     main()

from TSSmySQL import main as TSSmain
from RSSmySQL import main as RSSmain
from Ibpy import main as IBPYmain

import functools
from multiprocessing import Pool


def smap(f):
    return f()

def main():
    func1 = functools.partial(IBPYmain)
    func2 = functools.partial(TSSmain)
    func3 = functools.partial(RSSmain)

    with Pool() as pool:
        res = pool.map(smap, [func1, func2, func3])
        print(res)



    # put a while run loop and kill it when it hits market close
    # save todays on a hard copy on a csv in a folder and then clear the sql queries


if __name__ == '__main__':
    main()


