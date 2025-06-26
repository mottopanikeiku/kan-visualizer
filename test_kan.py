import torch
from kan_layer import KANLayer
from kan_network import KAN
from kan_trainer import KANTrainer


def test_kan_layer_creation():
    """test that kan layer can be created with valid params"""
    layer = KANLayer(in_features=5, out_features=3, grid_size=5)
    assert layer.in_features == 5
    assert layer.out_features == 3
    assert layer.grid_size == 5


def test_kan_layer_forward():
    """test kan layer forward pass with different input sizes"""
    layer = KANLayer(in_features=3, out_features=2, grid_size=5)
    
    # test with batch of inputs
    x = torch.randn(10, 3)
    output = layer(x)
    assert output.shape == (10, 2)
    
    # test with single input
    x = torch.randn(1, 3)
    output = layer(x)
    assert output.shape == (1, 2)


def test_kan_network_creation():
    """test that kan network can be created with valid architecture"""
    model = KAN(layers_hidden=[2, 5, 3, 1])
    assert len(model.layers) == 3  # 3 connections for 4 layers
    
    # check layer dimensions
    assert model.layers[0].in_features == 2
    assert model.layers[0].out_features == 5
    assert model.layers[1].in_features == 5
    assert model.layers[1].out_features == 3
    assert model.layers[2].in_features == 3
    assert model.layers[2].out_features == 1


def test_kan_network_forward():
    """test kan network forward pass"""
    model = KAN(layers_hidden=[2, 5, 1])
    x = torch.randn(8, 2)
    output = model(x)
    assert output.shape == (8, 1)


def test_kan_trainer_creation():
    """test that kan trainer can be created"""
    model = KAN(layers_hidden=[1, 5, 1])
    trainer = KANTrainer(model, optimizer_name="Adam", lr=0.01)
    assert trainer.model == model
    assert trainer.optimizer_name == "Adam"


def test_dataset_creation():
    """test synthetic dataset creation"""
    model = KAN(layers_hidden=[1, 5, 1])
    trainer = KANTrainer(model)
    
    def simple_func(x):
        return x ** 2
    
    x, y = trainer.create_dataset(
        func=simple_func,
        n_samples=100,
        input_dim=1,
        x_range=(-1, 1)
    )
    
    assert x.shape == (100, 1)
    assert y.shape == (100, 1)
    assert torch.all(x >= -1) and torch.all(x <= 1)


def test_simple_training():
    """test that training runs without errors"""
    # simple 1d function
    def target_func(x):
        return 2 * x + 1
    
    model = KAN(layers_hidden=[1, 3, 1], grid_size=3)
    trainer = KANTrainer(model, optimizer_name="Adam", lr=0.1)
    
    # small dataset for quick test
    x_train, y_train = trainer.create_dataset(
        func=target_func,
        n_samples=50,
        input_dim=1,
        x_range=(-1, 1)
    )
    
    # short training
    history = trainer.train(
        x_train, y_train,
        epochs=5,
        batch_size=10,
        verbose=False
    )
    
    assert len(history["train_loss"]) == 5
    assert all(isinstance(loss, float) for loss in history["train_loss"])


def test_regularization():
    """test that regularization loss computation works"""
    layer = KANLayer(in_features=2, out_features=1, grid_size=3)
    reg_loss = layer.regularization_loss(regularize_activation=0.1, regularize_entropy=0.1)
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.item() >= 0


def test_prediction():
    """test prediction functionality"""
    model = KAN(layers_hidden=[1, 3, 1])
    trainer = KANTrainer(model)
    
    x_test = torch.randn(5, 1)
    predictions = trainer.predict(x_test)
    
    assert predictions.shape == (5, 1)
    assert isinstance(predictions, torch.Tensor)


def test_different_activations():
    """test different base activation functions"""
    activations = ["silu", "relu", "gelu", "tanh"]
    
    for activation in activations:
        layer = KANLayer(
            in_features=2, 
            out_features=1, 
            base_activation=activation
        )
        x = torch.randn(3, 2)
        output = layer(x)
        assert output.shape == (3, 1)


if __name__ == "__main__":
    # run all tests
    test_functions = [
        test_kan_layer_creation,
        test_kan_layer_forward,
        test_kan_network_creation,
        test_kan_network_forward,
        test_kan_trainer_creation,
        test_dataset_creation,
        test_simple_training,
        test_regularization,
        test_prediction,
        test_different_activations
    ]
    
    print("Running KAN tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
    
    print("All tests completed!") 