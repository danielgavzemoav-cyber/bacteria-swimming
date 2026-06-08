import csv
import numpy as np
import matplotlib.pyplot as plt
MUM_IN_PIX = 0.225
SECS_PER_FRAME = 15.7 / 149


def pix2mum(pix):
    return pix * MUM_IN_PIX


def frame2time(frame):
    return frame * SECS_PER_FRAME


particles_tracks = {}  # {track_id: {'x_locations': [], 'y_locations': [], 'times': []}}
with open('exp_400_int_10_3_spots.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if row['TRACK_ID'].isdigit()]
    for row in rows:
        tid = int(row['TRACK_ID'])
        if tid not in particles_tracks:
            particles_tracks[tid] = {'radius': float(row['RADIUS']), 'area': float(row['AREA']), 'x_locations': [], 'y_locations': [], 'times': []}
        particles_tracks[tid]['x_locations'].append(pix2mum(float(row['POSITION_X'])))
        particles_tracks[tid]['y_locations'].append(pix2mum(float(row['POSITION_Y'])))
        particles_tracks[tid]['times'].append(frame2time(float(row['FRAME'])))

vs = []
radii = []
areas = []
omegas = []
plt.figure()
for track in particles_tracks:
    particles_tracks[track]['x_locations'] = np.array(particles_tracks[track]['x_locations'])
    particles_tracks[track]['y_locations'] = np.array(particles_tracks[track]['y_locations'])
    particles_tracks[track]['times'] = np.array(particles_tracks[track]['times'])
    indices = np.argsort(particles_tracks[track]['times'])
    x_locations = particles_tracks[track]['x_locations'][indices]
    y_locations = particles_tracks[track]['y_locations'][indices]
    times = particles_tracks[track]['times'][indices]
    plt.scatter(x_locations, y_locations, label=f'track: {track}')
    particles_tracks[track]['x_locations'] = x_locations
    particles_tracks[track]['y_locations'] = y_locations
    particles_tracks[track]['times'] = times
    # x_velocities = np.gradient(x_locations, times)
    # y_velocities = np.gradient(y_locations, times)
    x_velocities = np.diff(x_locations) / np.diff(times)
    y_velocities = np.diff(y_locations) / np.diff(times)
    velocities = np.sqrt((x_velocities ** 2) + (y_velocities ** 2))
    mean_velocity = np.mean(velocities)
    # x_rel_location = x_locations - x_locations[0]
    # y_rel_location = -(y_locations - y_locations[0])
    x_rel_location = x_locations - np.mean(x_locations)
    y_rel_location = -(y_locations - np.mean(y_locations))
    theta = np.arctan2(y_rel_location, x_rel_location)  # handles full circle (-π to π)
    theta_unwrapped = np.unwrap(theta)  # Unwrap the angle to avoid discontinuities at π ↔ -π
    omega = np.gradient(theta_unwrapped, times)  # Compute angular velocity ω = dθ/dt
    mean_angular_velocity = np.mean(omega)
    particles_tracks[track]['avg_v'] = mean_velocity
    particles_tracks[track]['velocities'] = velocities
    vs.append(mean_velocity)
    omegas.append(mean_angular_velocity)
    radii.append(particles_tracks[track]['radius'])
    areas.append(particles_tracks[track]['area'])
plt.legend()
plt.show(block=False)
vs = np.array(vs)
omegas = np.array(omegas)
plt.figure()
plt.hist(omegas, bins=30, color='skyblue', edgecolor='black')  # adjust bins as needed
plt.xlabel('Angular velocity $(rad/s)$')
plt.ylabel('Counts')
plt.title('Histogram of Angular Velocities - lower bound')
plt.grid(True)
plt.show(block=False)

plt.figure()
plt.scatter(radii, vs)
plt.xlabel('radii (in pixels)')
plt.ylabel('velocity $(\\mu m/s)$')
plt.grid(True)
plt.show(block=False)

plt.figure()
plt.scatter(areas, vs)
plt.xlabel('area (in pixels$^2$)')
plt.ylabel('velocity $(\\mu m/s)$')
plt.grid(True)
plt.show(block=False)

plt.figure()
plt.scatter(radii, omegas)
plt.xlabel('area (in pixels$^2$)')
plt.ylabel('angular velocity $(1/s)$')
plt.grid(True)
plt.show()