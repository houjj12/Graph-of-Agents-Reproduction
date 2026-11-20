import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "questions.json"


def build_math_question(rng, idx):
    a = rng.randint(6, 30)
    b = rng.randint(2, 12)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        correct = a + b
        stem = f"{a} + {b} 的结果是多少？"
    elif op == "-":
        correct = a - b
        stem = f"{a} - {b} 的结果是多少？"
    else:
        correct = a * b
        stem = f"{a} × {b} 的结果是多少？"

    wrong = set()
    while len(wrong) < 3:
        offset = rng.randint(1, 8)
        candidate = correct + rng.choice([-1, 1]) * offset
        if candidate != correct and candidate >= 0:
            wrong.add(candidate)

    options = [str(correct)] + [str(x) for x in wrong]
    rng.shuffle(options)
    return {
        "id": f"math-{idx}",
        "domain": "math",
        "stem": stem,
        "options": options,
        "answer": options.index(str(correct)),
    }


def build_fixed_questions(domain, rows):
    questions = []
    for idx, row in enumerate(rows, start=1):
        stem, options, correct = row
        questions.append(
            {
                "id": f"{domain}-{idx}",
                "domain": domain,
                "stem": stem,
                "options": options,
                "answer": options.index(correct),
            }
        )
    return questions


CODING = [
    ("Python 中定义函数使用哪个关键字？", ["func", "def", "function", "lambda"], "def"),
    ("Python 中获取列表长度使用哪个函数？", ["size()", "length()", "len()", "count()"], "len()"),
    ("Python 中单行注释使用哪个符号？", ["//", "#", "--", "/*"], "#"),
    ("Python 中输出内容通常使用哪个函数？", ["print()", "echo()", "write()", "output()"], "print()"),
    ("Python 中判断两个值是否相等使用哪个运算符？", ["=", "==", "===", "!="], "=="),
    ("Python 中创建空集合应使用哪个写法？", ["{}", "[]", "set()", "()"], "set()"),
    ("Python 中创建字典使用哪类数据结构？", ["list", "tuple", "set", "dict"], "dict"),
    ("Python 中遍历序列通常使用哪个关键字？", ["loop", "for", "while", "each"], "for"),
    ("Python 中逻辑“与”运算使用哪个关键字？", ["and", "or", "not", "xor"], "and"),
    ("Python 中导入模块使用哪个关键字？", ["include", "require", "import", "using"], "import"),
]

MEDICAL = [
    ("正常成年人的腋下体温大约是多少摄氏度？", ["35.0", "36.5", "37.8", "39.0"], "36.5"),
    ("成人安静时的正常心率范围通常是多少次每分钟？", ["40-50", "60-100", "110-140", "150-180"], "60-100"),
    ("人体进行气体交换的主要器官是什么？", ["肝脏", "肺", "胃", "肾脏"], "肺"),
    ("维生素 C 主要存在于哪类食物中？", ["肉类", "新鲜果蔬", "油炸食品", "奶制品"], "新鲜果蔬"),
    ("人体最大的器官是什么？", ["心脏", "肝脏", "皮肤", "肺"], "皮肤"),
    ("成年人血压测量中，收缩压的正常范围通常是多少 mmHg？", ["90-120", "50-70", "140-160", "180-200"], "90-120"),
    ("人体内负责运输氧气的主要细胞是什么？", ["白细胞", "红细胞", "血小板", "淋巴细胞"], "红细胞"),
    ("糖尿病患者通常需要控制哪种物质的摄入？", ["蛋白质", "糖类", "钙", "铁"], "糖类"),
    ("人体骨骼中，成年人共有约多少块骨？", ["126", "206", "306", "406"], "206"),
    ("哪种维生素与钙吸收密切相关？", ["维生素 A", "维生素 B", "维生素 C", "维生素 D"], "维生素 D"),
]

GENERAL = [
    ("地球绕太阳公转一周大约需要多长时间？", ["一天", "一个月", "一年", "十年"], "一年"),
    ("一年中哪个月份至少有 28 天？", ["一月", "二月", "三月", "每个月"], "每个月"),
    ("水的化学式是什么？", ["CO2", "H2O", "O2", "NaCl"], "H2O"),
    ("下列哪个国家位于亚洲？", ["巴西", "中国", "埃及", "澳大利亚"], "中国"),
    ("太阳从哪个方向升起？", ["西边", "南边", "东边", "北边"], "东边"),
    ("一周通常有多少天？", ["5", "6", "7", "8"], "7"),
    ("1 小时等于多少分钟？", ["30", "60", "100", "120"], "60"),
    ("三角形内角和是多少度？", ["90", "180", "270", "360"], "180"),
    ("世界上面积最大的海洋是哪一个？", ["大西洋", "印度洋", "太平洋", "北冰洋"], "太平洋"),
    ("“GDP”通常用于衡量什么？", ["人口数量", "国土面积", "国内生产总值", "空气质量"], "国内生产总值"),
]


def main():
    rng = random.Random(20261119)
    questions = []
    for idx in range(1, 11):
        questions.append(build_math_question(rng, idx))
    questions.extend(build_fixed_questions("coding", CODING))
    questions.extend(build_fixed_questions("medical", MEDICAL))
    questions.extend(build_fixed_questions("general", GENERAL))

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(questions)} questions -> {DATA_PATH}")


if __name__ == "__main__":
    main()
