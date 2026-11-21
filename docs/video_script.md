# Video script (5 minutes, camera on, whole screen shared)

The brief: state the problem, the agent's behaviour, the reward structure and the objective; run the best agent with the GUI and the terminal both visible, showing how you move through the folders and run the Python file; agent footage must fill at least 3 of the 5 minutes; finish on performance.

**Before recording:** close everything except one terminal (left half) and the browser (right half). Run `uv sync` once beforehand so nothing downloads on camera. Have Finder or `ls` ready to show the folders.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Camera + terminal in the repo folder | "This is my mission-based RL project: a medical delivery drone serving six rural health centres in Muhanga District, Rwanda. I trained four agents with TorchRL on the same environment." |
| 0:20–0:45 | `ls`, then `ls environment training models/pg` | "The environment is in `environment/custom_env.py`, training for DQN and the three policy-gradient methods is in `training/`, and the trained weights are in `models/`. Everything runs with `uv`." |
| 0:45–1:00 | Type `uv run main.py` and press Enter; the browser opens | **Problem:** "Requests for blood and emergency medicine arrive at random. Each has a deadline: if the drone misses it, the centre has to send a patient by road." |
| 1:00–1:30 | The viewer: point at the hub, a red beacon, the HUD | **Objective and actions:** "The agent is the hub's flight controller. Each decision it can fly to a centre and drop a package, return to the hub, load a package, recharge, or hold. Its goal is to deliver as many packages on time as possible, emergencies first, without running out of battery." |
| 1:30–2:00 | Point at the decision log and the terminal lines | **Reward:** "+10 for an emergency delivered on time, +4 for routine. −10 and −3 when a deadline is missed. A crash costs −40 plus every request it can no longer serve, and ending the shift away from the hub costs −15." |
| 2:00–4:15 | Let the agent fly (press F to follow the drone for a while, then orbit out) | **Behaviour, narrated as it happens:** "It loads two packages before leaving, so one trip serves two centres… here it recharges before the battery is low, because the next request could be 25 km away… here an emergency appears and it reroutes…" Read the terminal line for one delivery out loud. |
| 4:15–4:45 | Shift summary in the terminal; `assets/plots/generalisation.png` | **Performance:** "On 20 shifts it never trained on, the best agent scores [X], against [Y] for the hand-written dispatcher, and it still holds up on busier and windier days." |
| 4:45–5:00 | Camera | "The full sweeps, all forty runs and the analysis are in the repo and the report. Thank you." |

Agent footage runs from 0:45 to 4:15 (3.5 minutes), which clears the 3/4 rule. At the default speed one shift lasts about a minute, so the viewer will show three shifts; use `--episodes 4` if you want spare footage.
