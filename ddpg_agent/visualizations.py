from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
import math
import copy
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np
import pandas as pd

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
        action_gradients_im.set_data(q_a_frames.action_gradients.reshape(
            agent.q_a_frames_spec.ny, agent.q_a_frames_spec.nx))
        action_gradients_im.set_clim(q_a_frames.action_gradients.min(),
                                     q_a_frames.action_gradients.max())
        max_action_im.set_data(q_a_frames.max_action)
        actor_policy_im.set_data(q_a_frames.actor_policy.reshape(
            agent.q_a_frames_spec.ny, agent.q_a_frames_spec.nx))

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

labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']

# def visualize_agent(q_a_frame, title=""):
def visualize_quad_agent(agent, q_a_frames_spec=None):
    def get_cbar_ax(sgs):
        return fig.add_subplot(sgs.subgridspec(10,1)[1:-1])

    def show_current_pos(ax):
        state = agent.last_state
        ydim = q_a_frames_spec.y_dim
        xdim = q_a_frames_spec.x_dim
        # state=agent.transform_state(agent.last_state)
        state=agent.last_state
        # x=state[xdim]
        # y=state[ydim]
        x, y = fix_circular([state[xdim], state[ydim]])
        ax.plot(x, y, 'ro')

    def q_subplot(gs,im_arr,title):
        # TODO: Show current state as a point on plot
        ax = fig.add_subplot(gs)
        im = ax.pcolor(xs,ys,im_arr)
        show_current_pos(ax)
        cb = plt.colorbar(im,ax=ax,shrink=.8,pad=.02)
        ax.set_title(title)

    def gradients_subplot(gs, i, title=""):
        sgs=gs.subgridspec(1,12,wspace=.5)
        if i%2==0: # if on the left side put colorbar on the left
            cbax = get_cbar_ax(sgs[0])
            imax = fig.add_subplot(sgs[1:],xticks=[],yticks=[])
        else:
            cbax = get_cbar_ax(sgs[-1])
            imax = fig.add_subplot(sgs[:-1],xticks=[],yticks=[])
        if i==0: imax.set_title(" "*18+"Action Gradients")
        extr = max(np.abs(q_a_frame.action_gradients[:,:,i].min()),np.abs(q_a_frame.action_gradients[:,:,i].max()))
        im=imax.pcolor(xs,ys,q_a_frame.action_gradients[:,:,i],cmap='RdYlGn',vmin=-extr,vmax=extr)
        cb = plt.colorbar(im,ax=imax,cax=cbax)
        if i%2==0: cb.ax.yaxis.set_ticks_position('left')

    def policy_subplot(gs,i):
        sgs=gs.subgridspec(1,14,wspace=.4)
        if i%2==0: # if on the left side put colorbar on the left
            cbax = get_cbar_ax(sgs[0])
            ax = fig.add_subplot(sgs[1:],xticks=[],yticks=[])
        else:
            cbax = get_cbar_ax(sgs[-1])
            ax = fig.add_subplot(sgs[:-1],xticks=[],yticks=[])
        im=ax.pcolor(xs,ys,q_a_frame.actor_policy[:,:,i],
                     vmin=agent.env.action_space.low[i], vmax=agent.env.action_space.high[i]
                    )
        show_current_pos(ax)
        if i==0: ax.set_title(" "*42+"Policy")
        cb = plt.colorbar(im,ax=ax,cax=cbax)
        if i%2==0: cb.ax.yaxis.set_ticks_position('left')

    if q_a_frames_spec is None:
        q_a_frames_spec = agent.q_a_frames_spec
    q_a_frame = agent.get_q_a_frames(q_a_frames_spec)

    # title = "Agent at episode %i, step %i"%(
    #             agent.history.get_training_episode_for_step(q_a_frame.step_idx-1).episode_idx,
    #             q_a_frame.step_idx )

    xs = q_a_frames_spec.xs
    ys = q_a_frames_spec.ys
    fig = plt.figure(figsize=(13,7))
    # fig.suptitle(title)
    main_grid = gridspec.GridSpec(2, 4, figure=fig, wspace=.3, left=.05, right=.98)

    q_subplot(main_grid[0,0], q_a_frame.Q_max, title="Q max")
    q_subplot(main_grid[0,1], q_a_frame.Q_std, title="Q standard dev.")
    q_subplot(main_grid[1,0], q_a_frame.max_action, title="Action[%i] at Q max"%agent.q_a_frames_spec.a_dim)
    gradients_grid = main_grid[1,1].subgridspec(2,12,hspace=.02, wspace=.15)
    gradients_subplot(gradients_grid[0,:5], 0)
    gradients_subplot(gradients_grid[0,5:-2], 1)
    gradients_subplot(gradients_grid[1,:5], 2)
    gradients_subplot(gradients_grid[1,5:-2], 3)

    policy_grid = main_grid[:,2:].subgridspec(2,12,hspace=.02,wspace=.15)
    policy_subplot(policy_grid[0,:5], 0)
    policy_subplot(policy_grid[0,5:-2], 1)
    policy_subplot(policy_grid[1,:5], 2)
    policy_subplot(policy_grid[1,5:-2], 3)

    fig.show()

from collections import defaultdict, namedtuple
step_keys = ['step_idx','episode_idx',
               'state',
               'raw_action','action','reward','done',
               'x','y','z','phi','theta','psi',
               'x_velocity','y_velocity','z_velocity',
               'phi_velocity','theta_velocity','psi_velocity',
#                'rotor_speed1','rotor_speed2','rotor_speed3','rotor_speed4',
              ]
Step=namedtuple("Step", step_keys)

def fix_circular(arr):
    # Not really sure why this works, but "folding" negative values back onto the positive ones
    # fixes the visualizations of position angles.
    a=copy.copy(arr)
    for i in range(len(a)):
        if a[i]>math.pi: a[i]-=2*math.pi
    return a

def plot_quadcopter_episode(episode):
    fig = plt.figure(figsize=(15,7))
    fig.suptitle("Episode %i, score: %.3f, epsilon: %.4g"%(episode.episode_idx, episode.score, episode.epsilon))

    main_cols = gridspec.GridSpec(1, 3, figure=fig)
    right_col_grid = main_cols[1:].subgridspec(2,3,wspace=.2,hspace=.3)

    env_state=pd.DataFrame(episode.env_state)

    min_reward=min(episode.rewards)
    max_reward=max(episode.rewards)
    if min_reward==max_reward:
        min_reward=-1
        max_reward=1
    extreme=max(np.abs(min_reward),np.abs(max_reward))
    # reward_norm = mpl.colors.SymLogNorm(linthresh=1, linscale=3, vmin=-extreme, vmax=extreme)
    # reward_cmap = mpl.cm.ScalarMappable(norm=reward_norm, cmap=mpl.cm.get_cmap('RdYlGn'))
    reward_cmap = mpl.cm.ScalarMappable(
                            norm=mpl.colors.Normalize(vmin=-extreme, vmax=extreme),
                            cmap=mpl.cm.get_cmap('RdYlGn'))
    reward_cmap.set_array([])

    left_col = main_cols[0].subgridspec(3,1,hspace=.4)

    reward_ax = fig.add_subplot(left_col[0], title="Step Rewards")
    reward_ax.bar(range(len(episode.rewards)), episode.rewards,
                      color=[reward_cmap.to_rgba(r) for r in episode.rewards])
    #reward_ax.set_yscale("symlog")
    # reward_ax.set_ylim(-2, np.max(episode.rewards) )

    pos_ax = fig.add_subplot(left_col[1:], projection='3d', title="Flight Path")
    pos_scatter = pos_ax.scatter(env_state['x'], env_state['y'], env_state['z'],
                                 c=[reward_cmap.to_rgba(r) for r in episode.rewards],
                                 edgecolor='k', )

    fig.colorbar(reward_cmap, ax=pos_ax, shrink=.8, pad=.1, label="reward", orientation='horizontal')

    alt_ax = fig.add_subplot(right_col_grid[0,0], title="Altitude", xlabel='step')
    alt_ax.plot(env_state['z'], color='magenta')

    actions_grid = right_col_grid[0,2].subgridspec(4,1)
    def plot_action(i):
#         a_colors=['darkorange','darkgoldenrod','peru','lightsalmon']
        ax = fig.add_subplot(actions_grid[i], ylim=(-100,1000), xlabel='step', yticks=[0,400,800])
#         if i==0: ax.set_title('Actions')
        ax.plot([a[i] for a in episode.raw_actions], color='#1f77b4', label='policy action')
        ax.plot([a[i] for a in episode.actions], label='action + noise', color='red')#a_colors[i])
        if i==0: ax.legend(loc='lower center', bbox_to_anchor=(.5,.9))
    for i in range(4): plot_action(i)

    v_ax = fig.add_subplot(right_col_grid[1,0], title="Velocity", xlabel='step')
    v_ax.plot(env_state['x_velocity'], label="x")
    v_ax.plot(env_state['y_velocity'], label="y")
    v_ax.plot(env_state['z_velocity'], color='magenta', label="z")
    v_ax.legend(loc='upper right')

    rot_ax = fig.add_subplot(right_col_grid[1,1], title="Orientation", xlabel='step')
#     X=np.arange(len(step_arr)*3)/3
    rot_ax.plot(fix_circular(env_state['phi']), label='phi', color='#1f77b4', marker='.', lw=.1)
    rot_ax.plot(fix_circular(env_state['theta']), label='theta', color='#ff7f0e', marker='.', lw=.1)
    rot_ax.set_ylim(-math.pi, math.pi)
#     rot_ax.plot(phis[:,2], label='phi')
#     rot_ax.plot(thetas[:,2], label='theta')
#     #rot_ax.plot(X,psis.flatten(), color='darkgoldenrod', label='psi')
#     for i in range(3,5): plot_state(rot_ax,i)
    rot_ax.legend(loc='upper right')

    ang_v_ax = fig.add_subplot(right_col_grid[1,2], title="Angular Velocity", xlabel='step')
    ang_v_ax.plot(env_state['phi_velocity'], label="x")
    ang_v_ax.plot(env_state['theta_velocity'], label="y")
    ang_v_ax.plot(env_state['psi_velocity'], label="z")
    ang_v_ax.legend(loc='upper right')

    horiz_pos_ax = fig.add_subplot(right_col_grid[0,1], title="Horizontal Position")
    horiz_pos_ax.scatter(env_state['x'], env_state['y'],
                      c=[reward_cmap.to_rgba(r) for r in episode.rewards],
                      edgecolor='k',)
    # return plt.gcf()
    plt.show()

def plot_scores(training_scores=[], test_scores=[]):
    plt.title("Learning curve")
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.plot(training_scores, label='training')
    plt.plot(test_scores, label='test')
    plt.legend();
