import matplotlib.pyplot as plt



def input_image_visualizer(input_tensor):
    input_tensor=input_tensor.permute(1,2,0)
    input_tensor=input_tensor.round()
    image=plt.imshow(input_tensor)
    plt.tight_layout()
    plt.savefig("img.png",dpi=100)