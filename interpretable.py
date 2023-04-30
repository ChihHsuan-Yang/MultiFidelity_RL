def visualize_gradcam_latent_space(model, last_conv_layer_name, latent_space):
    # Convert the model with the custom output
    gradcam_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Create the Gradcam object
    gradcam = Gradcam(gradcam_model, model_modifier=None, clone=False)

    # Generate the heatmap for the input image
    cam = gradcam(latent_space, penultimate_layer=-1)
    cam = normalize(cam)

    # Render the heatmap
    heatmap = cm.jet(cam[0])[..., :3]

    return heatmap

# Usage example
latent_space = i_x.predict(train_image[0:1])  # Obtain the latent space of an image from your dataset
last_conv_layer_name = 'conv2d_12'  # The last convolutional layer in the x_j model

heatmap = visualize_gradcam_latent_space(x_j, last_conv_layer_name, latent_space)

# Display the heatmap
plt.imshow(heatmap.squeeze(), cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()


'''This function computes the heatmap for the latent space input using the Grad-CAM method and displays it as an image. 
Please adjust the last_conv_layer_name variable to match the name of the last convolutional layer in your x_j model.''''