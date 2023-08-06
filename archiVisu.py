import torch
from torchviz import make_dot

# Prepare a dummy input for the forward pass. I'm assuming that the
# input to your model is an image tensor of shape [3, H, W], where H, W are the image's height and width.
# You'll need to adjust the shape according to your specific requirements.
input_shape = (1, 3, 224, 224)  # Change 224x224 to the size of your images. The 1 is the batch size.
dummy_input = torch.randn(input_shape)

# Pass the dummy input through the image embedding and flatten the output
embedded_image = self.image_conv(dummy_input)
embedding_size = embedded_image.numel()  # total number of elements in the tensor
flattened_embedded_image = embedded_image.view(1, embedding_size)

# Perform a forward pass through the actor model and generate a visualization
actor_output = self.actor(flattened_embedded_image)
dot = make_dot(actor_output, params=dict(list(self.actor.named_parameters())))

# Save the result to a PDF
dot.format = 'pdf'
dot.render('actor_architecture')

# Repeat the process for the critic model
critic_output = self.critic(flattened_embedded_image)
dot = make_dot(critic_output, params=dict(list(self.critic.named_parameters())))

# Save the result to a PDF
dot.format = 'pdf'
dot.render('critic_architecture')
