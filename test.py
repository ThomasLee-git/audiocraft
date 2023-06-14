from audiocraft.models import MusicGen
import torchaudio


def test():
    # Using small model, better results would be obtained with `medium` or `large`.
    model = MusicGen.get_pretrained("melody")
    model.set_generation_params(use_sampling=True, top_k=250, duration=4)
    output = model.generate_unconditional(num_samples=1, progress=True)
    # torchaudio.save("")
    print(output.shape)


if __name__ == "__main__":
    test()
