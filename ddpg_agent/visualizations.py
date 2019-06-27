from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
import math
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

# plt.rcParams['animation.embed_limit'] = 200
plt.rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()

plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10

def create_animation(agent, every_n_steps=1, display_mode='gif', fps=30):
    history = agent.history
    fig = plt.figure(figsize=(11,6))
    fig.set_tight_layout(True)
    main_rows = gridspec.GridSpec(2, 1, figure=fig, top=.9, left=.05, right=.95, bottom=.25)

    def create_top_row_im(i, title='', actions_cmap=False):
        top_row = main_rows[0].subgridspec(1, 5, wspace=.3)
        ax = fig.add_subplot(top_row[i])
        ax.axis('off')
        ax.set_title(title)
        im = ax.imshow( np.zeros((len(history.q_a_frames_spec.ys),
                                  len(history.q_a_frames_spec.xs))), origin='lower' )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if actions_cmap is True:
            im.set_clim(history.q_a_frames_spec.amin,
                        history.q_a_frames_spec.amax)
            im.set_cmap("RdYlGn")
            cb = fig.colorbar(im, cax=cax)
        else:
            cb = fig.colorbar(im, cax=cax, format='%.3g')
        cb.ax.tick_params(labelsize=8)
        return im

    def create_bottom_row_plot(i, title=''):
        bottom_row = main_rows[1].subgridspec(1,3)
        ax = fig.add_subplot(bottom_row[i])
        ax.set_title(title)
        return ax

    Q_max_im = create_top_row_im(0, title='Q max')
    Q_std_im = create_top_row_im(1, title='Q standard deviation')
    action_gradients_im = create_top_row_im(2, title="Action Gradients")
    max_action_im = create_top_row_im(3, title="Action with Q max", actions_cmap=True)
    actor_policy_im = create_top_row_im(4, title="Policy", actions_cmap=True)

    scores_ax = create_bottom_row_plot(0, title="Scores")
    scores_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    scores_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    training_scores_line, = scores_ax.plot([], 'bo', label='training')
    test_scores_line, = scores_ax.plot([], 'ro', label='test')
    scores_ax.set_xlim(1,len(history.training_episodes))
    scores_combined = np.array([e.score for e in history.training_episodes ]+\
                               [e.score for e in history.test_episodes ])
    scores_ax.set_ylim(scores_combined.min(),scores_combined.max())
    scores_ax.set_xlabel('episode')
    scores_ax.set_ylabel('total reward')
    scores_ax.legend(loc='upper left', bbox_to_anchor=(0,-.1))

    training_episode_ax = create_bottom_row_plot(1)
    training_episode_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # TODO: get axis names from q_a_grid_spec
    training_episode_position_line, = training_episode_ax.plot([], 'b-', label='position')
    training_episode_velocity_line, = training_episode_ax.plot([], 'm-', label='velocity')
    training_episode_action_line, = training_episode_ax.plot([], 'r-', label='action')
    training_episode_reward_line, = training_episode_ax.plot([], 'g-', label='reward')
    training_episode_ax.set_ylim((-1.1,1.1))
    training_episode_ax.axes.get_yaxis().set_visible(False)
#     training_episode_ax.legend(loc='upper left', ncol=2)

    test_episode_ax = create_bottom_row_plot(2)
    test_episode_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    test_episode_position_line, = test_episode_ax.plot([], 'b-', label='position')
    test_episode_velocity_line, = test_episode_ax.plot([], 'm-', label='velocity')
    test_episode_action_line, = test_episode_ax.plot([], 'r-', label='action')
    test_episode_reward_line, = test_episode_ax.plot([], 'g-', label='reward')
    test_episode_ax.set_ylim((-1.1,1.1))
    test_episode_ax.axes.get_yaxis().set_visible(False)
    test_episode_ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(-.5,-.1))


    def update(step_idx):
        num_frames = math.ceil(last_step/every_n_steps)
        frame_idx = math.ceil(step_idx/every_n_steps)
        print("Drawing frame: %i/%i, %.2f%%\r"%\
              (frame_idx+1, num_frames, 100*(frame_idx+1)/float(num_frames) ), end='')
        training_episode = history.get_training_episode_for_step(step_idx)
        episode_step_idx = step_idx - training_episode.first_step

        q_a_frames = history.get_q_a_frames_for_step(step_idx)

        Q_max_im.set_data(q_a_frames.Q_max)
        Q_max_im.set_clim(q_a_frames.Q_max.min(),q_a_frames.Q_max.max())
        Q_std_im.set_data(q_a_frames.Q_std)
        Q_std_im.set_clim(q_a_frames.Q_std.min(),q_a_frames.Q_std.max())
        action_gradients_im.set_data(q_a_frames.action_gradients)
        action_gradients_im.set_clim(q_a_frames.action_gradients.min(),
                                     q_a_frames.action_gradients.max())
        max_action_im.set_data(q_a_frames.max_action)
        actor_policy_im.set_data(q_a_frames.actor_policy)

        # Plot scores
        xdata = range(1,training_episode.episode_idx+1)
        training_scores_line.set_data(xdata,
            [e.score for e in history.training_episodes ][:training_episode.episode_idx] )
        test_scores_line.set_data(xdata,
            [e.score for e in history.test_episodes ][:training_episode.episode_idx] )

        #Plot training episode
        training_episode_ax.set_title("Training episode %i, eps=%.3f, score: %.3f"%(
                            training_episode.episode_idx, training_episode.epsilon, training_episode.score))

        current_end_idx = episode_step_idx + every_n_steps
        if current_end_idx >= len(training_episode.states):
            current_end_idx = len(training_episode.states)-1

        training_xdata = range(0,current_end_idx+1)
        training_episode_ax.set_xlim(training_xdata[0],
                                     training_episode.last_step-training_episode.first_step+1)
        episode_states = [agent.preprocess_state(s) for s in training_episode.states]
        training_episode_position_line.set_data(training_xdata,
                                                [s[0] for s in episode_states][:current_end_idx+1])
        training_episode_velocity_line.set_data(training_xdata,
                                                [s[1] for s in episode_states][:current_end_idx+1])
        training_episode_action_line.set_data(training_xdata,
                                              training_episode.actions[:current_end_idx+1])
        training_episode_reward_line.set_data(training_xdata,
                                              training_episode.rewards[:current_end_idx+1])

        #Plot test episode
        test_episode = history.get_test_episode_for_step(step_idx)
        if test_episode is not None:
            test_episode_ax.set_title("Test episode %i, score: %.3f"%(
                                test_episode.episode_idx, test_episode.score))
            test_xdata = range(1,len(test_episode.states)+1)
            test_episode_ax.set_xlim(test_xdata[0],test_xdata[-1])
            episode_states = [agent.preprocess_state(e) for e in test_episode.states]
            test_episode_position_line.set_data(test_xdata, [s[0] for s in episode_states])
            test_episode_velocity_line.set_data(test_xdata, [s[1] for s in episode_states])
            test_episode_action_line.set_data(test_xdata,
                                              test_episode.actions )
            test_episode_reward_line.set_data(test_xdata, test_episode.rewards )

    last_step = history.training_episodes[-1].last_step + 1
    anim = FuncAnimation(fig, update, interval=1000/fps,
                         frames=range(0,last_step,every_n_steps))

    if display_mode=='video' or display_mode=='video_file':
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        if writer.isAvailable():
            print("Using ffmpeg at '%s'."%writer.bin_path())
        else:
            raise("FFMpegWriter not available for video output.")
    if display_mode=='js':
        display(HTML(anim.to_jshtml()))
    elif display_mode=='video':
        display(HTML(anim.to_html5_video()))
    elif display_mode=='video_file':
        filename = 'training_animation_%i.mp4'%int(datetime.now().timestamp())
        img = anim.save(filename, writer=writer)
        print("\rVideo saved to %s."%filename)
        # import io, base64
        # encoded = base64.b64encode(io.open(filename, 'r+b').read())
        # display(HTML(data='''<video alt="training animation" controls loop autoplay>
        #                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
        #              </video>'''.format(encoded.decode('ascii'))))
        display(HTML(data='''<video alt="training animation" controls loop autoplay>
                        <source src="{0}" type="video/mp4" />
                     </video>'''.format(filename)))
    else:
        filename = 'training_animation_%i.gif'%int(datetime.now().timestamp())
        img = anim.save(filename, dpi=80, writer='imagemagick')
        display(HTML("<img src='%s'/>"%filename))
    plt.close()
