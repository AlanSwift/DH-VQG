import os, pickle, json

def get():
    path = "/home/shiina/data/cocoqa/processed/train_split.pkl"
    with open(path, "rb") as f:
        split = pickle.load(f)
    answer_container = []
    for idx, inst in split.items():
        answer = inst["answer"]
        answer_container.append(answer)
    answer_list = list(set(answer_container))
    output = ""
    for answer in answer_list:
        output += answer + "\n"
    with open("answer.txt", "w") as f:
        f.write(output)

def make(answer_left_path, ann):
    with open(answer_left_path, "r") as f:
        answer_pool = f.readlines()
    with open(ann, "r") as f:
        cls2ans = json.load(f)
    ans2cls = {}
    ans2cls["one"] = "count"
    ans2cls["two"] = "count"
    ans2cls["three"] = "count"
    ans2cls["four"] = "count"
    ans2cls["five"] = "count"
    ans2cls["six"] = "count"
    ans2cls["seven"] = "count"
    ans2cls["eight"] = "count"
    ans2cls["nine"] = "count"
    ans2cls["ten"] = "count"
    ans2cls['cupcake'] = "food"
    ans2cls["station"] = "location"
    ans2cls["benches"] = "object"
    ans2cls["snowboards"] = "object"
    ans2cls["sailboat"] = "object"
    ans2cls["machine"] = "object"
    ans2cls["holder"] = "object"
    ans2cls["ramp"] = "object"
    ans2cls["towels"] = "object"
    ans2cls["cellphone"] = "object"
    ans2cls["dishes"] = "object"
    ans2cls["scooters"] = "object"
    ans2cls["surface"] = "spatial"
    ans2cls["parrot"] = "food"
    ans2cls["statues"] = "object"
    ans2cls["dogs"] = "animal"
    ans2cls["buildings"] = "object"
    ans2cls["pot"] = "object"
    ans2cls["appliances"] = "object"
    ans2cls["meal"] = "food"
    ans2cls["container"] = "object"
    ans2cls["outdoors"] = "location"
    ans2cls["screen"] = "object"
    ans2cls["painting"] = "activity"
    ans2cls["taxi"] = "object"
    ans2cls["suits"] = "object"
    ans2cls["wagon"] = "object"
    ans2cls["pans"] = "object"
    ans2cls["museum"] = "location"
    ans2cls["stadium"] = "location"
    ans2cls["bottles"] = "object"
    ans2cls["trolley"] = "object"
    ans2cls["kitty"] = "animal"
    ans2cls["case"] = "object"
    ans2cls["plants"] = "object"
    ans2cls["dock"] = "object"
    ans2cls["drawer"] = "object"
    ans2cls["cabinet"] = "object"
    ans2cls["hose"] = "object"
    ans2cls["shower"] = "activity"
    ans2cls["wheel"] = "object"
    ans2cls["bull"] = "animal"
    ans2cls["vehicles"] = "object"
    ans2cls["racquet"] = "object"
    ans2cls["bun"] = "food"
    ans2cls["trailer"] = "object"
    ans2cls["classroom"] = "location"
    ans2cls["freight"] = "object"
    ans2cls["gear"] = "object"
    ans2cls["cage"] = "object"
    ans2cls["tablet"] = "object"
    ans2cls["hallway"] = "location"
    ans2cls["toys"] = "object"
    ans2cls["trunk"] = "object"
    ans2cls["kitten"] = "location"
    ans2cls["room"] = "location"
    ans2cls["panda"] = "animal"
    ans2cls["highway"] = "location"
    ans2cls["tower"] = "object"
    ans2cls["slices"] = "food"
    ans2cls["rail"] = "object"
    ans2cls["pigeon"] = "animal"
    ans2cls["calf"] = "animal"
    ans2cls["track"] = "object"
    ans2cls["snack"] = "food"
    ans2cls["pool"] = "location"
    ans2cls["terminal"] = "location"
    ans2cls["wheelchair"] = "object"
    ans2cls["garage"] = "location"
    ans2cls["apartment"] = "object"
    ans2cls["helicopter"] = "object"
    ans2cls["backyard"] = "location"
    ans2cls["drink"] = "activity"
    ans2cls["helicopter"] = "object"
    ans2cls["marina"] = "location"
    ans2cls["pan"] = "object"
    ans2cls["tools"] = "object"
    ans2cls["vehicle"] = "object"
    ans2cls["pipe"] = "object"
    ans2cls["drinks"] = "object"
    ans2cls["lamb"] = "animal"
    ans2cls["ski"] = "activity"

    ans2cls["bar"] = "location"
    ans2cls["blender"] = "object"
    ans2cls["locomotive"] = "object"
    ans2cls["engine"] = "object"
    ans2cls["hill"] = "object"
    ans2cls["ram"] = "object"
    ans2cls["bath"] = "object"
    ans2cls["boards"] = "object"
    ans2cls["lane"] = "object"
    ans2cls["restroom"] = "location"
    ans2cls["bottle"] = "object"
    ans2cls["cigarette"] = "object"
    ans2cls["uniform"] = "object"
    ans2cls["monkey"] = "animal"
    ans2cls["pin"] = "object"
    ans2cls["device"] = "object"
    ans2cls["goats"] = "animal"
    ans2cls["mask"] = "object"
    ans2cls["sofa"] = "object"
    ans2cls["airliner"] = "object"
    ans2cls["trail"] = "object"
    ans2cls["carriage"] = "object"
    ans2cls["fridge"] = "object"
    ans2cls["dryer"] = "object"
    ans2cls["stall"] = "object"
    ans2cls["vest"] = "object"
    ans2cls["pie"] = "food"
    ans2cls["store"] = "location"
    ans2cls["television"] = "object"
    ans2cls["duck"] = "animal"
    ans2cls["pony"] = "animal"
    ans2cls["shuttle"] = "object"
    ans2cls["dish"] = "food"
    ans2cls["shop"] = "location"
    ans2cls["pug"] = "animal"
    ans2cls["scene"] = "object"
    ans2cls["biplane"] = "object"
    ans2cls["subway"] = "object"
    ans2cls["foil"] = "object"
    ans2cls["pastries"] = "food"
    ans2cls["photograph"] = "object"
    ans2cls["can"] = "object"
    ans2cls["stick"] = "object"
    ans2cls["seat"] = "object"
    ans2cls["tram"] = "object"
    ans2cls["paw"] = "object"
    ans2cls["apron"] = "object"
    ans2cls["owl"] = "animal"
    ans2cls["grizzly"] = "animal"
    ans2cls["bucket"] = "object"
    ans2cls["eagle"] = "object"
    ans2cls["urinals"] = "object"
    ans2cls["cap"] = "object"
    ans2cls["puppy"] = "animal"
    ans2cls["tub"] = "object"
    ans2cls["cart"] = "object"
    ans2cls["frame"] = "object"
    ans2cls["warehouse"] = "location"
    ans2cls["furniture"] = "object"
    ans2cls["bank"] = "location"
    ans2cls["sculpture"] = "object"
    ans2cls["urinal"] = "object"
    ans2cls["beverage"] = "object"
    ans2cls["curtain"] = "object"
    ans2cls["cattle"] = "object"
    ans2cls["crib"] = "object"
    ans2cls["hydrant"] = "object"
    ans2cls["hangar"] = "object"
    ans2cls["garden"] = "object"
    ans2cls["candle"] = "object"
    ans2cls["equipment"] = "object"
    ans2cls["suit"] = "object"
    ans2cls["hillside"] = "location"
    ans2cls["bathtub"] = "object"
    ans2cls["rack"] = "object"
    ans2cls["slice"] = "food"
    ans2cls["vegetable"] = "food"
    ans2cls["jar"] = "food"
    ans2cls["platter"] = "object"
    ans2cls["library"] = "location"
    ans2cls["walkway"] = "location"
    ans2cls["goat"] = "animal"
    ans2cls["fireplace"] = "object"
    ans2cls["statue"] = "object"
    ans2cls["barn"] = "object"
    ans2cls["pen"] = "object"
    ans2cls["tray"] = "object"
    ans2cls["scooter"] = "object"
    ans2cls["doorway"] = "location"
    ans2cls["entree"] = "object"
    ans2cls["crosswalk"] = "location"
    ans2cls["driveway"] = "location"
    ans2cls["tent"] = "object"
    ans2cls["pastry"] = "food"
    ans2cls["shelves"] = "object"
    ans2cls["hamburger"] = "food"
    ans2cls["side"] = "object"
    ans2cls["freezer"] = "object"
    ans2cls["canoe"] = "object"
    ans2cls["coat"] = "object"
    ans2cls["booth"] = "object"
    ans2cls["ship"] = "object"
    ans2cls["dress"] = "object"
    ans2cls["ingredients"] = "object"
    ans2cls["toy"] = "object"
    ans2cls["paddle"] = "object"




    print(cls2ans.keys())
    for cls, ans_list in cls2ans.items():
        for ans in ans_list:
            ans2cls[ans] = cls
    cnt = 0
    for ans in answer_pool:
        ans = ans.strip()
        if ans2cls.get(ans) is None:
            if len(ans) > 1:
                if ans[-2:] == "es":
                    prefix_es = ans[:-2]
                    if ans2cls.get(prefix_es) is not None:
                        ans2cls[ans] = ans2cls[prefix_es]
                        cnt += 1
                        continue

                if ans[-1] == "s":
                    prefix_s = ans[:-1]
                    if ans2cls.get(prefix_s) is not None:
                        ans2cls[ans] = ans2cls[prefix_s]
                        cnt += 1
                        continue
                print(ans)
                continue

        else:
            cnt += 1
    print(cnt, len(answer_pool))
    exit(0)
    cocoqa_cls = {}
    for ans, cls in ans2cls.items():
        if cocoqa_cls.get(cls) is None:
            cocoqa_cls[cls] = []
        cocoqa_cls[cls].append(ans)
    with open("cocoqa_cls.json", "w") as f:
        json.dump(cocoqa_cls, f)


if __name__ == "__main__":
    make("answer.txt", "iq.json")