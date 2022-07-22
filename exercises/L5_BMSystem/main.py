import bminf
import sys


def generate(model: bminf.models.CPM1, sentence):
    print("Input: ", sentence)
    sys.stdout.write("Output: %s" % sentence)
    stoped = False
    while not stoped:
        value, stoped = model.generate(
            sentence,
            max_tokens=8,
            top_n=5,
            top_p=None,
            temperature=0.85,
            frequency_penalty=0,
            presence_penalty=1
        )
        sentence += value
        sys.stdout.write(value)
        sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    print("Loading model")
    cpm1 = bminf.models.CPM1()
    print("Start")
    while True:
        input_text = input("菜名：")
        generate(cpm1, input_text + "的做法是：")


if __name__ == "__main__":
    main()
