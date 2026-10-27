"""Batched point forces. Contact layout [body_id, local_xyz, world_Fxyz, mask].

MuJoCo xfrc_applied is [world force, world torque ABOUT BODY COM], not link origin.
The stored contact is held over a control interval; lever arm is recomputed each substep.
"""
import torch

CONTACTS = 3
FORCE_DIM = CONTACTS * 8
TEACHER_DIM = FORCE_DIM + 3


def features(contacts):
    scale = contacts.new_tensor([32., .3, .3, .3, 30., 30., 30., 1.])
    return ((contacts / scale) * contacts[..., 7:8]).flatten(-2)


def teacher_features(delayed, memory):
    return torch.cat((features(delayed), memory), dim=-1)


def update_memory(memory, applied):
    # Causal low-pass force history, never a task label or selected destination.
    return .995 * memory + .005 * applied[..., 4:7].mul(applied[..., 7:8]).sum(-2) / 30.


def latch(target, contacts):
    # Reward-only desired endpoint (0 standing, 1 squatting).
    fz = (contacts[..., 6] * contacts[..., 7]).sum(-1)
    return torch.where(fz < -3., torch.ones_like(target), torch.where(fz > 3., torch.zeros_like(target), target))


def pulse(step, onset, duration, magnitude, ramp=25):
    t = step - onset
    a = (t.float() / ramp).clamp(0, 1)
    b = ((duration - t).float() / ramp).clamp(0, 1)
    return magnitude * a * a * (3 - 2 * a) * b * b * (3 - 2 * b)


def scheduled_contacts(step, schedule, torso):
    # [onset, duration, signed_magnitude, second_onset]; optional opposite second pulse.
    fz = pulse(step, schedule[..., 0], schedule[..., 1], schedule[..., 2])
    fz += pulse(step, schedule[..., 3], schedule[..., 1], -schedule[..., 2])
    out = torch.zeros(step.shape + (CONTACTS, 8), device=step.device)
    out[..., 0, 0] = torso
    out[..., 0, 3] = .15
    out[..., 0, 6] = fz
    out[..., 0, 7] = (fz.abs() > 0).float()
    return out


def wrench_from_contacts(contacts, xpos, xmat, xipos):
    ids = contacts[..., 0].long()
    batch = torch.arange(contacts.shape[0], device=contacts.device)[:, None]
    rot = xmat.reshape(xmat.shape[0], xmat.shape[1], 3, 3)[batch, ids]
    point = xpos[batch, ids] + (rot @ contacts[..., 1:4, None]).squeeze(-1)
    force = contacts[..., 4:7] * contacts[..., 7:8]
    torque = torch.linalg.cross(point - xipos[batch, ids], force, dim=-1)
    result = xpos.new_zeros(xpos.shape[:2] + (6,))
    result.scatter_add_(1, ids[..., None].expand(-1, -1, 6), torch.cat((force, torque), -1))
    return result


def apply_contacts(sim, contacts):
    d = sim.data
    d.xfrc_applied[:] = wrench_from_contacts(contacts, d.xpos, d.xmat, d.xipos)
