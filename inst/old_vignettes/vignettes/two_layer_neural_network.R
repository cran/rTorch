## ---- include = FALSE----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ------------------------------------------------------------------------
library(rTorch)

device = torch$device('cpu')

# device = torch.device('cuda') # Uncomment this to run on GPU

## ------------------------------------------------------------------------
torch$manual_seed(0)

N <- 64L; D_in <- 1000L; H <- 100L; D_out <- 10L

# Create random Tensors to hold inputs and outputs
x = torch$randn(N, D_in, device=device)
y = torch$randn(N, D_out, device=device)


## ------------------------------------------------------------------------
model = torch$nn$Sequential(
          torch$nn$Linear(D_in, H),
          torch$nn$ReLU(),
          torch$nn$Linear(H, D_out))$to(device)

## ------------------------------------------------------------------------
loss_fn = torch$nn$MSELoss(reduction = 'sum')

## ------------------------------------------------------------------------
learning_rate = 1e-4

for (t in 1:500) {
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the loss.
  loss = loss_fn(y_pred, y)
  
  cat(t, "\t")
  cat(loss$item(), "\n")
  
  # Zero the gradients before running the backward pass.
  model$zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Tensors with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss$backward()

  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its data and gradients like we did before.
  with(torch$no_grad(), {
      for (param in iterate(model$parameters())) {
        # in Python this code is much simpler. In R we have to do some conversions
        
        # param$data <- torch$sub(param$data,
        #                         torch$mul(param$grad$float(),
        #                           torch$scalar_tensor(learning_rate)))
        
        param$data <- param$data - param$grad * learning_rate
      }
   })
}  

