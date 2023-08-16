import os
import requests

discriminator = './RVC/pretrained_v2/f0D48k.pth'
generator = './RVC/pretrained_v2/f0G48k.pth'
hubert = './RVC/hubert_base.pt'

def get_model():
    if not os.path.isfile(discriminator) or os.path.isfile(generator) or os.path.isfile(hubert):
        url1 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/f0D48k.pth'
        url2 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/f0G48k.pth'
        url3 = 'https://github.com/ORI-Muchim/BARK-RVC/releases/download/v1.0/hubert_base.pt'

        print("Downloading Pretrained Discriminator Model...")
        response1 = requests.get(url1, allow_redirects=True)

        print("Downloading Pretrained Generator Model...")
        response2 = requests.get(url2, allow_redirects=True)
        
        print("Downloading Hubert Model...")
        response3 = requests.get(url3, allow_redirects=True)

        directory = './RVC/pretrained_v2'
        directory2 = './RVC'

        pretrained_discriminator_model = os.path.join(directory, 'f0D48k.pth')
        pretrained_generator_model = os.path.join(directory, 'f0G48k.pth')
        hubert_model = os.path.join(directory2, 'hubert_base.pt')

        with open(pretrained_discriminator_model, 'wb') as file:
            file.write(response1.content)
        print("Saving Pretrained Discriminator Model...")

        with open(pretrained_generator_model, 'wb') as file:
            file.write(response2.content)
        print("Saving Pretrained Generator Model...")
        
        with open(hubert_model, 'wb') as file:
            file.write(response3.content)
        print("Saving Hubert Model...\n")
    else:
        print('Skipping Download... Model exists.')