import torch
import torch.nn as tnn

import match
import match.nn as mnn


def test_linear():
    match.manual_seed(42)
    torch.manual_seed(42)

    in_features, out_features = 4, 3

    m_linear = mnn.Linear(in_features, out_features)
    t_linear = tnn.Linear(in_features, out_features)

    # Align weights and biases for identical comparison
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(m_linear.weight.data.tolist()).T)
        t_linear.bias.copy_(torch.tensor(m_linear.bias.data.tolist()).squeeze(0))

    x_data = [[1.0, 2.0, -1.0, 0.5], [0.5, -0.5, 1.5, 2.0]]
    m_x = match.tensor(x_data, requires_grad=True)
    t_x = torch.tensor(x_data, requires_grad=True)

    m_out = m_linear(m_x)
    t_out = t_linear(t_x)

    assert m_out.shape == t_out.shape
    for m_val, t_val in zip(m_out.data.data, t_out.detach().flatten().tolist()):
        assert abs(m_val - t_val) < 1e-4

    m_loss = m_out.sum()
    t_loss = t_out.sum()

    m_loss.backward()
    t_loss.backward()

    for m_g, t_g in zip(m_linear.weight.grad.data, t_linear.weight.grad.T.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4

    for m_g, t_g in zip(m_linear.bias.grad.data, t_linear.bias.grad.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4


def test_activations():
    x_data = [[1.5, -2.0], [-0.5, 3.0]]

    # ReLU
    m_x = match.tensor(x_data, requires_grad=True)
    t_x = torch.tensor(x_data, requires_grad=True)
    m_relu = mnn.ReLU()(m_x)
    t_relu = tnn.ReLU()(t_x)
    m_relu.sum().backward()
    t_relu.sum().backward()
    for m_val, t_val in zip(m_relu.data.data, t_relu.detach().flatten().tolist()):
        assert abs(m_val - t_val) < 1e-4
    for m_g, t_g in zip(m_x.grad.data, t_x.grad.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4

    # Sigmoid
    m_x = match.tensor(x_data, requires_grad=True)
    t_x = torch.tensor(x_data, requires_grad=True)
    m_sig = mnn.Sigmoid()(m_x)
    t_sig = tnn.Sigmoid()(t_x)
    m_sig.sum().backward()
    t_sig.sum().backward()
    for m_val, t_val in zip(m_sig.data.data, t_sig.detach().flatten().tolist()):
        assert abs(m_val - t_val) < 1e-4
    for m_g, t_g in zip(m_x.grad.data, t_x.grad.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4


def test_mse_loss():
    input_data = [[1.0, 2.0], [3.0, 4.0]]
    target_data = [[1.5, 1.8], [2.8, 4.2]]

    m_input = match.tensor(input_data, requires_grad=True)
    t_input = torch.tensor(input_data, requires_grad=True)
    m_target = match.tensor(target_data)
    t_target = torch.tensor(target_data)

    m_loss = mnn.MSELoss()(m_input, m_target)
    t_loss = tnn.MSELoss()(t_input, t_target)

    assert abs(m_loss.item() - t_loss.item()) < 1e-4

    m_loss.backward()
    t_loss.backward()

    for m_g, t_g in zip(m_input.grad.data, t_input.grad.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4


def test_cross_entropy_loss():
    logits_data = [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]
    target_indices = [0, 1]

    m_logits = match.tensor(logits_data, requires_grad=True)
    t_logits = torch.tensor(logits_data, requires_grad=True)
    m_target = match.tensor(target_indices)
    t_target = torch.tensor(target_indices, dtype=torch.long)

    m_loss = mnn.CrossEntropyLoss()(m_logits, m_target)
    t_loss = tnn.CrossEntropyLoss()(t_logits, t_target)

    assert abs(m_loss.item() - t_loss.item()) < 1e-4

    m_loss.backward()
    t_loss.backward()

    for m_g, t_g in zip(m_logits.grad.data, t_logits.grad.flatten().tolist()):
        assert abs(m_g - t_g) < 1e-4
