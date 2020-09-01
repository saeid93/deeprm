import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"
import sys
import getopt
import matplotlib
import json
matplotlib.use('Agg')

import parameters
import pg_re
import pg_su
import slow_down_cdf


"""
parameters read from config.json:

exp_type:              <type of experiment>
num_res:               <number of resources>
num_nw:                <number of visible new work>
simu_len:              <simulation length>
num_ex:                <number of examples>
num_seq_per_batch:     <rough number of samples in one batch update>
eps_max_len:           <episode maximum length (terminated at the end)>
num_epochs:            <number of epoch to do the training>
time_horizon:          <time step into future, screen height> 
res_slot:              <total number of resource slots, screen width> 
max_job_len:           <maximum new job length>
max_job_size:          <maximum new job resource request>
new_job_rate:          <new job arrival rate>
dist:                  <discount factor>
lr_rate:               <learning rate>
ba_size:               <batch size>
pg_re:                 <parameter file for pg network>
v_re:                  <parameter file for v network>
q_re:                  <parameter file for q network>
out_freq:              <network output frequency>
ofile:                 <output file name>
log:                   <log file name>
render:                <plot dynamics>
unseen:                <generate unseen example>
"""

def main():
    config_file = "config.json"
    with open(config_file) as cf:
        arg = json.loads(cf.read()) # TODO arg to config
    
    pa = parameters.Parameters()


    type_exp = arg['exp_type']
    pa.num_res = arg['num_res']
    pa.num_nw = arg['num_nw']
    pa.simu_len = arg['simu_len']
    pa.num_ex = arg['num_ex']
    pa.num_seq_per_batch = arg['num_seq_per_batch']
    pa.episode_max_length = arg['eps_max_len']
    pa.num_epochs = arg['num_epochs']
    pa.time_horizon = arg['time_horizon']
    pa.res_slot = arg['res_slot']
    pa.max_job_len = arg['max_job_len']
    pa.max_job_size = arg['max_job_size']
    pa.new_job_rate = arg['new_job_rate']
    pa.discount = arg['dist']
    pa.lr_rate = arg['lr_rate']
    pa.batch_size = arg['ba_size']
    pg_resume = arg['pg_re']
    v_resume = arg['v_re']
    q_resume = arg['q_re']
    pa.output_freq = arg['out_freq']
    pa.output_filename = arg['ofile']
    log = arg['log']
    render = arg['render']
    pa.generate_unseen = arg['unseen'] # TODO what is the use


# TODO debug and find out what compute_dependent_parameters() does

    pa.compute_dependent_parameters()

    if type_exp == 'pg_su':
        pg_su.launch(pa, pg_resume, render, repre='image', end='all_done')
    elif type_exp == 'v_su':
        v_su.launch(pa, v_resume, render)
    elif type_exp == 'pg_re':
        pg_re.launch(pa, pg_resume, render, repre='image', end='all_done')
    elif type_exp == 'pg_v_re':
        pg_v_re.launch(pa, pg_resume, v_resume, render)
    elif type_exp == 'test':
        # quick_test.launch(pa, pg_resume, render)
        slow_down_cdf.launch(pa, pg_resume, render, True)
    # elif type_exp == 'q_re':
    #     q_re.launch(pa, q_resume, render)
    else:
        print("Error: unkown experiment type " + str(type_exp))
        exit(1)


if __name__ == '__main__':
    main()