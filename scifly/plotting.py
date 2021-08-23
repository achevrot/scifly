# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_pickle("/mnt/meso/Data/Testing_data/new_crash/autoencoder/ath_lon_2021/0_4CA9D1.pkl")
df['colour'] = pd.get_dummies(df.phase).values.argmax(1)
df['x'] = range(len(df))

fig, ax = plt.subplots()
fig.set_dpi(300)

df2 = df.loc[:3100]

fakeLine1 = plt.Line2D([0,0],[0,1], color='green')
fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')
fakeLine3 = plt.Line2D([0,0],[0,1], color='orange')

color = ['green', 'blue', 'orange']
legend = ['CLIMB', 'CRUISE', 'DESCENT']

for i in range(3):
    data = df2.loc[df2['colour'] == i]
    plt.scatter('x', 'altitude', data=data, color=color[i], label=legend[i], s=0.5)
ax.legend([fakeLine1,fakeLine2,fakeLine3], ['CLIMB', 'CRUISE', 'DESCENT'])
ax.set_xlabel('message number')
ax.set_ylabel('altitude')
plt.show()


# %%

#Plotting Hijack DAE

fig, ax = plt.subplots()
fig.set_dpi(300)

fakeLine1 = plt.Line2D([0,0],[0,1], color='purple')
fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')
fakeLine3 = plt.Line2D([0,0],[0,1], color='red')


plt.hlines(y=0.10656501600294294, xmin=0, xmax=389,colors = 'purple',)
plt.vlines(x = 389, ymin = 0.07988857754414519, ymax = 0.10656501600294294, colors = 'purple') 
plt.hlines(y=0.07988857754414519, xmin=389, xmax=1920,colors = 'purple',)
plt.vlines(x = 1920, ymin = 0.07988857754414519, ymax = 0.11254387042295558, colors = 'purple') 
plt.hlines(y=0.11254387042295558, xmin=1920, xmax=3000,colors = 'purple',)

plt.plot(test_score_df[0])

ax.legend([fakeLine1,fakeLine2, fakeLine3], ['anomaly threshold', 'anomaly score', 'Anomaly start'])

ax.set_xlabel('message number')
ax.set_ylabel('anomaly score')
plt.grid(alpha=0.4)

plt.vlines(x = 1067, ymin = 0, ymax = 0.6, 
           colors = 'red', 
           label = 'Injected anomaly') 

ax.yaxis.set_ticks(np.arange(0, 0.6, 0.05))
ax.xaxis.set_ticks(np.arange(0, 1800, 200))
ax.set_ylim([0,0.6])
ax.set_xlim([0,1600])

plt.show()

# %%

#Plotting Hijack AE

fig, ax = plt.subplots()
fig.set_dpi(300)

fakeLine1 = plt.Line2D([0,0],[0,1], color='purple')
fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')
fakeLine3 = plt.Line2D([0,0],[0,1], color='red')


# plt.hlines(y=0.11742360144853592, xmin=0, xmax=449,colors = 'purple',)
# plt.vlines(x = 449, ymin = 0.055783714167773724, ymax = 0.11742360144853592, colors = 'purple') 
plt.hlines(y=0.0223817, xmin=2000, xmax=3400,colors = 'purple',)
# plt.vlines(x = 3185, ymin = 0.07988857754414519, ymax = 0.11254387042295558, colors = 'purple') 
# plt.hlines(y=0.11254387042295558, xmin=3185, xmax=3600,colors = 'purple',)

plt.plot(test_score_df["distance"][2000:])

ax.legend([fakeLine1,fakeLine2, fakeLine3], ['anomaly threshold', 'anomaly score', 'emergency ON'], loc='upper left',)

ax.set_xlabel('message number')
ax.set_ylabel('anomaly score')
plt.grid(alpha=0.4)

plt.vlines(x = 3205, ymin = 0, ymax = 0.1, 
           colors = 'red', 
           label = 'emergency ON') 

ax.yaxis.set_ticks(np.arange(0, 0.2, 0.05))
ax.xaxis.set_ticks(np.arange(2000, 3557, 200))

plt.show()

# %%

fig, ax = plt.subplots()
fig.set_dpi(300)

fakeLine1 = plt.Line2D([0,0],[0,1], color='purple')
fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')
fakeLine3 = plt.Line2D([0,0],[0,1], color='red')


# plt.hlines(y=0.11742360144853592, xmin=0, xmax=449,colors = 'purple',)
# plt.vlines(x = 449, ymin = 0.055783714167773724, ymax = 0.11742360144853592, colors = 'purple') 
plt.hlines(y=0.09194569662213326, xmin=0, xmax=2000,colors = 'purple',)
# plt.vlines(x = 3185, ymin = 0.055783714167773724, ymax = 0.11083418503403664, colors = 'purple') 
# plt.hlines(y=0.11083418503403664, xmin=3185, xmax=3000,colors = 'purple',)

plt.plot(test_score_df['test_mae_loss'])

ax.legend([fakeLine1,fakeLine2, fakeLine3], ['anomaly threshold', 'anomaly score', 'emergency ON'])

ax.set_xlabel('message number')
ax.set_ylabel('anomaly score')
plt.grid(alpha=0.4)

plt.vlines(x = 928, ymin = 0, ymax = 0.6, 
           colors = 'red', 
           label = 'emergency ON') 

ax.yaxis.set_ticks(np.arange(0, 0.65, 0.05))
ax.xaxis.set_ticks(np.arange(0, 2000, 100))

plt.show()

#%%

original_df = pd.read_csv("/mnt/meso/Data/Testing_data/world_data/FDIT/train_set/lon_mil_2021/15_471f7a.bst", header=None)
altered_df = pd.read_csv("/mnt/meso/Data/Testing_data/pos_deviation/FDIT/train_ds_21/lon_mil_2021/15_471f7a/15_471f7a_1806813400_0.bst", header=None)

fig, ax = plt.subplots()
fig.set_dpi(300)


fakeLine1 = plt.Line2D([0,0],[0,1], color='orange')
fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')

# fakeLine1 = plt.Line2D([0,0],[0,1], color='purple')
# fakeLine2 = plt.Line2D([0,0],[0,1], color='blue')
# fakeLine3 = plt.Line2D([0,0],[0,1], color='red')


# plt.hlines(y=0.055783714167773724, xmin=2500, xmax=3185,colors = 'purple',)
# plt.hlines(y=0.11083418503403664, xmin=3185, xmax=4000,colors = 'purple',)
# plt.vlines(x = 3185, ymin = 0.055783714167773724, ymax = 0.11083418503403664, 
#            colors = 'purple', 
#            label = 'vline_multiple - full height') 

ax.legend([fakeLine1,fakeLine2], ['original data', 'attacked data'])

plt.plot(altered_df[15], altered_df[14])
plt.plot(original_df[15], original_df[14])
# plt.plot(data_raw.groundspeed)

# ax.legend([fakeLine1,fakeLine2, fakeLine3], ['anomaly threshold', 'anomaly score', 'emergency ON'])

ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
# plt.grid(alpha=0.4)

# plt.vlines(x = 3133, ymin = 0, ymax = 0.6, 
#            colors = 'red', 
#            label = 'emergency ON') 

# ax.yaxis.set_ticks(np.arange(0, 0.65, 0.05))

plt.grid(alpha=0.4)
# ax.yaxis.set_ticks(np.arange(0, 2000, 200))

plt.show()

# %%

mean_asc[mean_asc >0.5] = mean_asc.mean()
mean_cru[mean_cru >0.5] = mean_cru.mean()
mean_des[mean_des >0.5] = mean_des.mean()

# %%

fig, ax = plt.subplots()
fig.set_dpi(300)

fakeLine1 = plt.Line2D([0,0],[0,1], color='orange')
fakeLine2 = plt.Line2D([0,0],[0,1], color='green')
fakeLine3 = plt.Line2D([0,0],[0,1], color='blue')

plt.vlines(x = 0.10656501600294294, ymin = 0, ymax = 600, colors = 'orange') 
plt.vlines(x = 0.07988857754414519, ymin = 0, ymax = 600, colors = 'green') 
plt.vlines(x = 0.11254387042295558, ymin = 0, ymax = 600, colors = 'blue') 

plt.hist(mean_asc, bins='auto', alpha=0.5, color='orange') # bleu
plt.hist(mean_cru, bins='auto', alpha=0.5, color='green') # orange
plt.hist(mean_des, bins='auto', alpha=0.5, color='blue') # vert
ax.set_xlim(0,0.2)

ax.legend([fakeLine1,fakeLine2, fakeLine3], ['CLIMB', 'CRUISE', 'DESCENT'])

ax.set_xlabel('anomaly score')
ax.set_ylabel('frequency')

# ax.xaxis.set_ticks(np.arange(0, 0.6, 0.05))
plt.show()
