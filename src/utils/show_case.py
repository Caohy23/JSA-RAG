import json
import math
path = 'atlas_data/experiments/jsa-test-4-2/training_info_step1.json'
T=2
training_info = json.load(open(path, 'r'))
query = training_info['query'].replace('question: ', '').replace(' answer: <extra_id_0>', '')
answer = training_info['response'].replace('<extra_id_0> ', '')
passages = training_info['Retrieved_top20_passages']
prior_probs = [float(item) for item in training_info['Prior_probs'].split(',')]
post_probs = [float(item) for item in training_info['Post_probs'].split(',')]
proposed_ids = training_info['Proposal_indices']
proposed_lm_probs = [float(item) for item in training_info['Proposed_log_lm_probs'].split(',')]
random_numbers = [float(item) for item in training_info['Random_numbers'].split(',')]
for i in range(len(proposed_lm_probs)):
    print(f'*******************************step {i+1}*******************************')
    print('Query:', query)
    print('Answer:', answer)
    print('Proposed passage:', passages[proposed_ids[i]])
    if i==0:
        print('prior prob:', prior_probs[proposed_ids[i]])
        print('posterior prob:', post_probs[proposed_ids[i]])
        print('log lm prob:', proposed_lm_probs[i])
        r = prior_probs[proposed_ids[i]]/post_probs[proposed_ids[i]] 
        print('weight: {:.3f}*exp({})'.format(r, proposed_lm_probs[i]))
        print('directly accept (the first step)')
        pv_r = r
        pv_log_lm_prob = proposed_lm_probs[i]
    else:
        print('prior prob:', prior_probs[proposed_ids[i]])
        print('posterior prob:', post_probs[proposed_ids[i]])
        print('log lm prob:', proposed_lm_probs[i])
        r = prior_probs[proposed_ids[i]]/post_probs[proposed_ids[i]] 
        print('weight: {:.3f}*exp({})'.format(r, proposed_lm_probs[i]))
        accept_rate = math.exp((proposed_lm_probs[i]-pv_log_lm_prob)/T)*r/pv_r
        print('accept rate:', accept_rate)
        print('random number:', random_numbers[i-1])
        if random_numbers[i-1]<=accept_rate:
            print('accept')
            pv_r = r
            pv_log_lm_prob = proposed_lm_probs[i]
        else:
            print('reject')
    input()

