import matplotlib, os, errno
# IF WE ARE ON SERVER WITH NO DISPLAY, then we use Agg:
#print matplotlib.get_backend()
if not('DISPLAY' in os.environ):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def visualize_history(hi, show=True, save=False, save_path='', show_also='', custom_title=None):
    # Visualize history of Keras model run.
    '''
    Example calls:
    hi = model.fit(...)
    saveHistory(hi.history, 'tmp_saved_history.npy')
    visualize_history(loadHistory('tmp_saved_history.npy'))
    '''

    # list all data in history
    print(hi.keys())
    # summarize history for loss
    plt.plot(hi['loss'])
    plt.plot(hi['val_loss'])

    if show_also is not '':
        plt.plot(hi[show_also], linestyle='dotted')
        plt.plot(hi['val_'+show_also], linestyle='dotted')

    if custom_title is None:
        plt.title('model loss')
    else:
        plt.title(custom_title)

    plt.ylabel('loss')
    plt.xlabel('epoch')

    if show_also == '':
        plt.legend(['train', 'test'], loc='best')
    else:
        plt.legend(['train', 'test', 'train-'+show_also, 'test-'+show_also], loc='best')


    if save:
        filename = save_path #+'loss.png'
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                #if exc.errno != errno.EEXIST:
                #    raise
                print("couldn't make a folder", filename)

        plt.savefig(filename)
        plt.savefig(filename+'.pdf', format='pdf')

        print("Saved image to "+filename)
    if show:
        plt.show()

    #plt.clf()
    return plt

def visualize_histories(histories, show=True, save=False, save_path='', show_also='', custom_title=None):
    for hi in histories[:-1]:
        visualize_history(hi, show=False, save=False, save_path='', show_also=show_also, custom_title=None)

    plt = visualize_history(histories[-1], show=show, save=save, save_path=save_path, show_also=show_also, custom_title=custom_title)
    plt.clf()
    return 0

def visualize_special_histories(histories, plotvalues='loss', show=True, save=False, save_path='', custom_title=None, just_val=False):
    '''
    We are visualizing results of a k-fold crossvalidation training. In <histories> we have the individual runs of the experiment.
    '''

    train_color = 'grey'
    val_color = 'blue'

    avg_train_color = 'red'
    avg_val_color = 'green'

    avg_train = []
    avg_val = []

    # count the averages

    epochs = len(histories[0][plotvalues])
    for epoch in range(0, epochs):
        trains = []
        vals = []
        for hi in histories:
            train = hi[plotvalues][epoch]
            val = hi['val_'+plotvalues][epoch]
            trains.append(train)
            vals.append(val)
        avg_train.append( np.mean(trains) )
        avg_val.append( np.mean(vals) )

    import matplotlib.pyplot as plt
    plt.figure()

    if custom_title is None:
        custom_title = 'model ' + plotvalues
    if just_val:
        custom_title = custom_title + ' (just validation results)'

    i = 0
    leg = []
    if not just_val:
        leg.append('average training')
    leg.append('average validation')

    if not just_val:
        leg.append('training errors')
    leg.append('validation errors')

    # now averages:
    if not just_val:
        plt.plot(avg_train, color=avg_train_color)
    plt.plot(avg_val, color=avg_val_color)

    for hi in histories:
        # list all data in history
        print(hi.keys())
        # summarize history for loss
        if not just_val:
            plt.plot(hi[plotvalues], linestyle='dashed', color=train_color)
        plt.plot(hi['val_'+plotvalues], linestyle='dashed', color=val_color)
        i += 1

    # OK, but we also want these on top...:
    if not just_val:
        plt.plot(avg_train, color=avg_train_color)
    plt.plot(avg_val, color=avg_val_color)


    plt.title(custom_title)
    plt.ylabel('loss')
    plt.xlabel('epoch')


    #plt.legend(leg, loc='lower left')
    plt.legend(leg, loc='best')
    if save:
        plt.savefig(save_path) #+plotvalues+'.png')
        plt.savefig(save_path+'.pdf', format='pdf')

    if show:
        plt.show()

    plt.clf()
    return plt

