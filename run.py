import json
from ui import setup

config = json.load(open("config.json", "r"))

if not config["ui_test"]:
    from model import IFModel

    model = IFModel(**config)
    demo = setup(model)
else:
    demo = setup()

demo.queue()
demo.launch(share=config["share"])
