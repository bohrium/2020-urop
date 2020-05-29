''' author: jason and sam
    change: 2020-05-29
    create: 2020-05-29
    descrp: 
    to use: Specify a slurm*.err dreamcoder log as a command line argument.
            For example, run:

                python plot_accs.py  slurm.node003.16960939.err  voila.png

            and find the output in voila.png
'''

import matplotlib.pyplot as plt
import sys
import random 

#=============================================================================#
#=====  0. PARSE a DREAMCODER LOG  ===========================================#
#=============================================================================#

def collect_stats(chunk):
    ''' From a list of hit/miss lines culled from one iteration of a dreamcoder
        log, produce a dictionary of accuracies as well as a set of hits.  The
        dictionary has type ( wave -> concept -> float ) and the hits are a
        subset of ( concept ).
    '''
    counts = {}
    hits = {}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.0. For each Parsed Line...  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    for line in chunk:
        status, test_case = line.split()[:2] 
        wave, concept, sample_index = test_case.split('_')  

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  0.1. ... Update Count of Hits vs Total  ~~~~~~~~~~~~~~~~~~~~~~#

        if wave not in counts: counts[wave] = {} 
        if concept not in counts[wave]: counts[wave][concept] = [0, 0]
        if status == 'HIT':
            counts[wave][concept][0] += 1
        counts[wave][concept][1] += 1

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  0.2. ... and Remember the Hits  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        if wave not in hits: hits[wave] = set()
        if status == 'HIT':
            hits[wave].add(concept)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.3. Finish by Dividing Hits by Totals  ~~~~~~~~~~~~~~~~~~~~~~#

    accs_by_wave = {
        wave: {
            concept: float(hits)/total
            for concept, (hits, total) in v.items()
        } for wave, v in counts.items()
    } 
    return accs_by_wave, hits

#=============================================================================#
#=====  1. PARSING AND COUNTING  =============================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.0. How Many Concepts have High Accuracies?  ~~~~~~~~~~~~~~~~#

def fraction_solved(accs, acc_pred): 
    ''' compute the fraction of the given dictionary's entries whose values
        obey the given predicate
    '''
    return (
        float(len([a for a in accs.values() if acc_pred(a)])) /
        len(accs)
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.1. Which Concepts were Recently Cracked / Still Unsolved?  ~#

def find_cracked(accs_a, accs_b, acc2_pred = lambda a, b: a < b): 
    ''' sample uniformly from concepts for which progress was recently made '''
    diff = {k for k in accs_a if acc2_pred(accs_a[k], accs_b[k])} 
    return random.choice(list(diff)) if diff else None 

def find_unsolved(accs, acc_pred = lambda a: 0.0 != a): 
    ''' sample uniformly from concepts on which no progress has been made '''
    unsolved = {k for k in accs if not acc_pred(accs[k])}
    return random.choice(list(unsolved)) if unsolved else None 

#=============================================================================#
#=====  2. MAIN LOOP  ========================================================#
#=============================================================================#

if __name__ == '__main__':

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.0. Read Text from File  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    file_name, image_name = sys.argv[1:][:2]
    print('reading file {}'.format(file_name))
    with open(file_name) as f:
        text = f.read()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.1. Parse Text into One Chunk per Epoch  ~~~~~~~~~~~~~~~~~~~~#

    chunks = text.split('Generative model enumeration results:')[1:]
    chunks = [c.split('Hits')[0] for c in chunks] 
    chunks = [[line for line in c.split('\n') if line] for c in chunks] 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.2. Collect Statistics into Iteration-Indexed Lists  ~~~~~~~~#
   
    waves = ('wave1', 'wave3')
    accs = {w: [] for w in waves}
    hits = {w: [] for w in waves}
    for c in chunks:
        accs_by_wave, hits_by_wave = collect_stats(c)
        for w in waves:
            accs[w].append(accs_by_wave[w])
            hits[w].append(hits_by_wave[w])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.3. Plot each Curve  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
    preds_by_name = {
        'solved completely': (lambda a: a == 1.0, '.'),
        'solved partially':  (lambda a: 0.0 != a, 'x'),
    }
    
    for w in waves:
        for pred_name, (pred, m) in preds_by_name.items():
            curve = [fraction_solved(accs, pred) for accs in accs[w]]
            plt.plot(curve, marker = m,
                     label = '{} tasks {}'.format(w, pred_name))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.4. Display Representative Concepts  ~~~~~~~~~~~~~~~~~~~~~~~~#

    for w in waves:
        curve = [fraction_solved(accs, lambda x: x!=0) for accs in accs[w]]
        for i in range(len(chunks)):
            hard_concept = find_unsolved(accs[w][i]) 
            if hard_concept is not None:
                plt.text(i, curve[i]-0.01, hard_concept, rotation=-45, color='k')

            if i == 0: continue

            fresh_concept = find_cracked(accs[w][i-1], accs[w][i])
            if fresh_concept is not None:
                plt.text(i-1.0, curve[i]+0.00, fresh_concept, rotation=0, color='g')

    plt.scatter([], [], color='g', label='sample concepts cracked')
    plt.scatter([], [], color='k', label='sample concepts unsolved')
 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.5. Label and Save Plot  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    plt.legend()
    plt.ylim([0.0, 0.4])
    plt.title('DreamCoder\'s Learning Curves on Rule List Tasks')
    plt.xlabel('Number of iterations')
    plt.ylabel('Percentage of tasks solved')

    print('saving to {}'.format(image_name))
    plt.savefig(image_name)
