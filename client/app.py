"""
Big Five Personality — Streamlit UI (ported from app.html).
Inference: masked Ridge bundle at models/artifacts/big5_ridge.joblib (see models/usage.ipynb).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "big5_ridge.joblib"

TRAITS: dict[str, dict[str, str]] = {
    "EXT": {"name": "Extraversion", "th": "ความเปิดเผย"},
    "EST": {"name": "Neuroticism", "th": "ความวิตกกังวล"},
    "AGR": {"name": "Agreeableness", "th": "ความเป็นมิตร"},
    "CSN": {"name": "Conscientiousness", "th": "ความมีระเบียบ"},
    "OPN": {"name": "Openness", "th": "ความเปิดรับ"},
}

RATING_CAPS = ["ไม่ใช่เลย", "ไม่ค่อยใช่", "กลางๆ", "ค่อนข้างใช่", "ใช่มากเลย"]

MEME_META: dict[str, dict[str, str]] = {
    "CSN6": {
        "label": "ขี้ลืมเรื่องของ",
        "desc": "วางของแล้วจำไม่ได้ว่าวางไว้ที่ไหน บ่อยแค่ไหนกัน?",
        "low": "แทบไม่เลย",
        "high": "บ่อยมากกก",
    },
    "AGR9": {
        "label": "อ่านอารมณ์คนเก่ง",
        "desc": "รู้สึกได้เลยว่าคนรอบข้างกำลังรู้สึกอะไรอยู่",
        "low": "ไม่ค่อยรู้เลย",
        "high": "รู้ทุกอย่างเลย",
    },
    "AGR5": {
        "label": "ไม่แคร์ปัญหาคนอื่น",
        "desc": "ปัญหาของคนอื่นมันไม่ใช่เรื่องของเราสักหน่อย",
        "low": "แคร์มากเลย",
        "high": "ไม่แคร์เลย",
    },
    "EXT4": {
        "label": "ชอบหลบอยู่หลังๆ",
        "desc": "ไม่ชอบเป็นจุดสนใจ ขอนั่งเงียบๆ ดีกว่า",
        "low": "ชอบโดดเด่น",
        "high": "ขอหลบแถวหลัง",
    },
    "EST7": {
        "label": "อารมณ์แปรปรวน",
        "desc": "อารมณ์เปลี่ยนเร็วมาก บางทีตัวเองก็งงว่าทำไม",
        "low": "นิ่งมากๆ",
        "high": "แปรปรวนมากๆ",
    },
    "OPN3": {
        "label": "จินตนาการเยอะ",
        "desc": "ในหัวมีภาพ มีเรื่องราว มีโลกส่วนตัวอยู่ตลอด",
        "low": "แทบไม่มีเลย",
        "high": "เยอะมากๆ",
    },
    "OPN10": {
        "label": "ไอเดียพุ่งพล่าน",
        "desc": "คิดเรื่องใหม่ๆ ตลอด บางทีคิดไม่หยุดเลย",
        "low": "แทบไม่มี",
        "high": "มีตลอดเวลา",
    },
}

# Must match `meme_questions` in `big5_ridge.joblib` (used when artifact missing).
MEME_QIDS_FALLBACK: tuple[str, ...] = (
    "CSN6",
    "AGR9",
    "AGR5",
    "EXT4",
    "EST7",
    "OPN3",
    "OPN10",
)

ALL_QUESTIONS: dict[str, tuple[str, str]] = {
    "EXT1": ("EXT", "เวลาไปปาร์ตี้ คนมักวนมาหาเราเป็นจุดศูนย์กลาง"),
    "EXT2": ("EXT", "ปกติไม่ค่อยพูดอะไรมาก"),
    "EXT3": ("EXT", "อยู่กับคนอื่นแล้วรู้สึกสบายใจดี"),
    "EXT4": ("EXT", "ชอบอยู่แถวหลังๆ ไม่ต้องเป็นจุดสนใจก็ได้"),
    "EXT5": ("EXT", "มักเป็นคนเริ่มต้นสนทนาก่อนเสมอ"),
    "EXT6": ("EXT", "บางทีนึกไม่ออกว่าจะพูดอะไร"),
    "EXT7": ("EXT", "งานปาร์ตี้นี่คุยได้กับทุกคนเลย"),
    "EXT8": ("EXT", "ไม่อยากให้ใครมาจับตามองเราหรอก"),
    "EXT9": ("EXT", "ถ้าต้องเป็นจุดสนใจก็ไม่ได้รู้สึกอะไรมาก"),
    "EXT10": ("EXT", "เจอคนแปลกหน้ามักจะเงียบกริบ"),
    "EST1": ("EST", "เครียดง่ายมาก นิดนึงก็เครียดแล้ว"),
    "EST2": ("EST", "ส่วนใหญ่รู้สึกผ่อนคลายดี ไม่ค่อยตึงเครียด"),
    "EST3": ("EST", "ชอบกังวลเรื่องต่างๆ อยู่เรื่อยๆ"),
    "EST4": ("EST", "ไม่ค่อยรู้สึกหดหู่หรือซึมเศร้า"),
    "EST5": ("EST", "อารมณ์ถูกกระทบได้ง่ายมาก"),
    "EST6": ("EST", "หัวร้อนง่าย โกรธเร็ว"),
    "EST7": ("EST", "อารมณ์เปลี่ยนบ่อย บางทีตัวเองก็งง"),
    "EST8": ("EST", "อารมณ์แปรปรวนบ่อยๆ ไม่นิ่ง"),
    "EST9": ("EST", "หงุดหงิดง่ายมาก"),
    "EST10": ("EST", "มักรู้สึกหดหู่หรือเหนื่อยใจบ่อยๆ"),
    "AGR1": ("AGR", "ไม่ค่อยสนใจคนอื่นเท่าไหร่"),
    "AGR2": ("AGR", "สนใจชีวิตและเรื่องราวของคน"),
    "AGR3": ("AGR", "บางทีก็ชอบแซวหรือดูถูกคนอื่นเล่นๆ"),
    "AGR4": ("AGR", "เข้าใจความรู้สึกของคนอื่นได้ดี"),
    "AGR5": ("AGR", "ปัญหาของคนอื่นไม่ใช่เรื่องของเรา"),
    "AGR6": ("AGR", "เป็นคนใจอ่อน ใจดี"),
    "AGR7": ("AGR", "แทบไม่แคร์ว่าคนรอบข้างเป็นยังไง"),
    "AGR8": ("AGR", "พร้อมเสียสละเวลาช่วยคนอื่นเสมอ"),
    "AGR9": ("AGR", "รู้สึกได้ถึงอารมณ์ของคนรอบข้าง"),
    "AGR10": ("AGR", "ทำให้คนอื่นรู้สึกสบายใจได้เก่ง"),
    "CSN1": ("CSN", "เตรียมของพร้อมอยู่เสมอ ไม่ค่อยลืมอะไร"),
    "CSN2": ("CSN", "ของมักกระจัดกระจายไปทั่ว"),
    "CSN3": ("CSN", "ใส่ใจรายละเอียดเล็กๆ น้อยๆ เสมอ"),
    "CSN4": ("CSN", "ที่ไหนที่อยู่มักรกหน่อยๆ"),
    "CSN5": ("CSN", "งานบ้านนี่ทำเสร็จทันทีเลย ไม่ทิ้งค้างไว้"),
    "CSN6": ("CSN", "วางของแล้วมักจำไม่ได้ว่าวางไว้ที่ไหน"),
    "CSN7": ("CSN", "ชอบความเป็นระเบียบเรียบร้อย"),
    "CSN8": ("CSN", "มักหลีกเลี่ยงหน้าที่ที่ต้องรับผิดชอบ"),
    "CSN9": ("CSN", "ใช้ชีวิตตามตารางเวลาที่วางไว้"),
    "CSN10": ("CSN", "ทำอะไรก็ทำอย่างละเอียดรอบคอบ"),
    "OPN1": ("OPN", "ใช้คำศัพท์หลากหลาย คิดคำได้เยอะ"),
    "OPN2": ("OPN", "เรื่องนามธรรมหรือแนวคิดซับซ้อนนี่เข้าใจยากหน่อย"),
    "OPN3": ("OPN", "จินตนาการแจ่มมาก ภาพในหัวชัดเจน"),
    "OPN4": ("OPN", "ไม่ค่อยสนใจแนวคิดหรือทฤษฎีนามธรรม"),
    "OPN5": ("OPN", "มีไอเดียดีๆ บ่อยมาก"),
    "OPN6": ("OPN", "ไม่ค่อยมีจินตนาการเท่าไหร่"),
    "OPN7": ("OPN", "เข้าใจเรื่องยากๆ ได้ไวกว่าคนอื่น"),
    "OPN8": ("OPN", "ชอบใช้คำศัพท์ยากๆ ดูฉลาด"),
    "OPN9": ("OPN", "ชอบนั่งคิดทบทวนเรื่องต่างๆ อยู่คนเดียว"),
    "OPN10": ("OPN", "ในหัวมีไอเดียตลอดเวลา หยุดคิดไม่ได้"),
}

TRAIT_COLORS = {
    "EXT": "#7f77dd",
    "EST": "#d85a30",
    "AGR": "#1d9e75",
    "CSN": "#378add",
    "OPN": "#ba7517",
}

# ---------------------------------------------------------------------------
# Paragraph copy — 9 slots per trait, each slot covers 0.5 of the 1–5 scale.
# slot_idx = min(8, int((score - 1) / 0.5))
# slot 0 = 1.0–1.5 … slot 8 = 5.0
# Each entry: (short_tag, paragraph)
# ---------------------------------------------------------------------------
TRAIT_PARAGRAPHS: dict[str, list[tuple[str, str]]] = {
    "EXT": [
        ("เก็บตัวสุดๆ",
         "คุณได้พลังงานจากการอยู่คนเดียวเต็มๆ โลกข้างในมีความสุขและความหมายให้คุณมากกว่าการสังสรรค์ คนรอบข้างอาจดูคุณเงียบ แต่นั่นคือวิธีที่คุณอยู่ได้ดีที่สุด"),
        ("เก็บตัวมาก",
         "การอยู่กับตัวเองคือการชาร์จแบตจริงๆ คุณเลือกสังคมอย่างระมัดระวัง ไม่ใช่เพราะขี้อาย แต่เพราะรู้ว่าพลังงานของตัวเองมีค่า"),
        ("ค่อนข้างเก็บตัว",
         "คุณไม่ได้กลัวคน แค่ไม่ได้หิวสังคมเท่าคนอื่น ปาร์ตี้ใหญ่ดูน่าเหนื่อย แต่คุยกับเพื่อนสนิทสักคนสองคนนั้นโอเคมาก"),
        ("กลางๆ เอียงเก็บตัว",
         "คุณอยู่กับคนอื่นได้สบายๆ แต่ก็ต้องการเวลาคนเดียวเป็นระยะๆ ถ้าสังคมเยอะเกินไปติดกันหลายวัน คุณจะรู้สึกเหนื่อยนิดนึง"),
        ("กลางๆ",
         "คุณยืดหยุ่นได้ดีมาก บางวันก็อยากออกไปเจอคน บางวันก็ขอนอนอยู่บ้าน ขึ้นอยู่กับอารมณ์และสถานการณ์มากกว่าสไตล์ตายตัว"),
        ("กลางๆ เอียงเปิดเผย",
         "คุณชอบอยู่กับคนอื่นและมักรู้สึกดีขึ้นหลังออกสังคม แต่ก็ไม่ได้รู้สึกแย่ถ้าต้องอยู่คนเดียวบ้าง"),
        ("ค่อนข้างเปิดเผย",
         "คุณสนุกกับการพบปะผู้คน เริ่มต้นบทสนทนาได้ไม่ยาก และมักเป็นคนที่ทำให้บรรยากาศในกลุ่มดีขึ้น"),
        ("เปิดเผยมาก",
         "คุณชาร์จแบตจากการอยู่กับคนอื่น ยิ่งมีคนเยอะยิ่งสนุก คุณเป็นคนที่ดึงพลังงานมาให้ทั้งวง และมักเป็นคนที่ทุกคนจำได้"),
        ("ขวัญใจมหาชน",
         "คุณเกิดมาเพื่ออยู่กับผู้คน การสังสรรค์คือการเติมพลัง ไม่ใช่การสิ้นเปลือง คุณรู้สึกมีชีวิตชีวาที่สุดเมื่ออยู่ท่ามกลางฝูงชน"),
    ],
    "EST": [
        ("นิ่งสงบมาก",
         "อารมณ์คุณมั่นคงมาก แทบไม่มีอะไรทำให้คุณหวั่นไหวได้ง่ายๆ คนรอบข้างมักพึ่งพาคุณในยามที่ทุกอย่างวุ่นวาย เพราะรู้ว่าคุณไม่ตื่นตระหนก"),
        ("อารมณ์มั่นคง",
         "คุณรับมือกับความเครียดได้ดีกว่าคนส่วนใหญ่ เมื่อเจอปัญหาคุณมักมองหาทางออกก่อน ไม่ใช่จมอยู่กับความรู้สึก"),
        ("ค่อนข้างนิ่ง",
         "คุณไม่ได้เครียดง่าย แต่ก็ไม่ได้หุ่นยนต์ เจอเรื่องใหญ่จริงๆ ก็รู้สึกอยู่ แต่ไม่นานก็กลับมาสู่สมดุลได้เอง"),
        ("กลางๆ เอียงนิ่ง",
         "คุณมักจัดการอารมณ์ได้ดี แต่มีบางสถานการณ์ที่กังวลหรือเครียดได้บ้าง ซึ่งก็เป็นเรื่องปกติมาก"),
        ("กลางๆ",
         "คุณรู้สึกกับสิ่งต่างๆ ได้ปกติ บางครั้งก็เครียดบางครั้งก็โอเค ขึ้นอยู่กับเหตุการณ์มากกว่าอุปนิสัยตายตัว"),
        ("กลางๆ เอียงกังวล",
         "คุณรู้สึกถึงความกดดันได้ชัด และบางครั้งก็วนคิดเรื่องเดิมอยู่นาน แต่ก็ผ่านมาได้ในที่สุด"),
        ("ค่อนข้างกังวล",
         "คุณรับรู้และประมวลอารมณ์ได้ลึก ซึ่งทำให้คุณเห็นอกเห็นใจคนอื่นได้มาก แต่ก็หมายความว่าความเครียดสะสมได้ง่ายถ้าไม่ระวัง"),
        ("กังวลบ่อย",
         "อารมณ์ของคุณเคลื่อนไหวเร็ว คุณรู้สึกทุกอย่างอย่างเต็มที่ ซึ่งเป็นพรและความท้าทายในเวลาเดียวกัน การดูแลตัวเองเป็นสิ่งสำคัญมากสำหรับคุณ"),
        ("อารมณ์เข้มข้น",
         "คุณมีชีวิตอารมณ์ที่สมบูรณ์และเข้มข้นมาก รู้สึกดีใจได้สุดขั้วและเศร้าได้ลึก การหาพื้นที่ปลอดภัยให้กับตัวเองเป็นเรื่องสำคัญมาก"),
    ],
    "AGR": [
        ("ตรงไปตรงมามาก",
         "คุณพูดสิ่งที่คิดโดยไม่ค่อยกรองผ่านเลนส์ความรู้สึกของคนอื่น ไม่ใช่เรื่องเลว — คุณแค่ให้ความจริงมากกว่าความสบายใจ"),
        ("ยึดจุดยืนตัวเอง",
         "คุณไม่ได้ใส่ใจว่าคนอื่นจะรู้สึกยังไงกับความคิดเห็นของคุณมาก สิ่งที่คิดว่าถูกต้องคือสิ่งที่จะพูด"),
        ("มีจุดยืนชัด",
         "คุณแคร์คนบ้าง แต่ไม่ได้ยอมตามทุกอย่างเพื่อรักษาความสัมพันธ์ คุณเลือกว่าเรื่องไหนควรยืน เรื่องไหนควรยืดหยุ่น"),
        ("กลางๆ เอียงตรง",
         "คุณมีน้ำใจแต่ก็มีขอบเขต ไม่ยอมทุกอย่างแต่ก็ไม่ได้แข็งกระด้าง คนส่วนใหญ่รู้สึกสบายใจพอที่จะคุยกับคุณได้"),
        ("สมดุลดี",
         "คุณแคร์คนอื่นและก็รักษาจุดยืนตัวเองได้พอๆ กัน ไม่ได้โอนอ่อนทุกอย่าง แต่ก็ไม่ได้แข็งจนคนเข้าไม่ถึง"),
        ("กลางๆ เอียงอ่อนโยน",
         "คุณมักพยายามทำให้ทุกคนรู้สึกดี และมักประนีประนอมได้ง่าย บางทีอาจเก็บความรู้สึกตัวเองไว้แทนที่จะพูดออกมา"),
        ("ใจอ่อนใจดี",
         "คุณเข้าใจความรู้สึกคนอื่นได้ดีมากและมักเอาใจเขามาใส่ใจเรา แต่อย่าลืมดูแลความต้องการของตัวเองด้วยนะ"),
        ("เอาใจใส่สูง",
         "คุณรู้สึกได้ถึงอารมณ์ของคนรอบข้างเกือบจะทันที และมักจะทำทุกอย่างเพื่อช่วย ทักษะนี้ทรงพลังมาก แต่ก็เหนื่อยได้ถ้าไม่มีขอบเขต"),
        ("เอาใจคนเก่งมาก",
         "คุณอ่านคนออกเก่งมาก รู้ว่าใครต้องการอะไรก่อนที่เขาจะพูด คุณทำให้คนรอบข้างรู้สึกว่าตัวเองมีคุณค่าและถูกเห็น"),
    ],
    "CSN": [
        ("ชีวิตอิสระมาก",
         "คุณใช้ชีวิตตามสัญชาตญาณ ไม่ผูกตัวเองกับตารางหรือแผนงานมากเกินไป บางคนอาจเรียกว่าไม่มีระเบียบ แต่คุณเรียกมันว่าอิสรภาพ"),
        ("ยืดหยุ่นสูง",
         "คุณปรับตัวตามสถานการณ์ได้ดีมาก แต่บางครั้งก็อาจลืมนัดหรือทำอะไรค้างไว้บ้าง ซึ่งไม่ได้ทำให้คุณไม่ดี แค่ไม่ได้เน้นโครงสร้างมาก"),
        ("ค่อนข้างยืดหยุ่น",
         "คุณมีระเบียบในเรื่องที่สำคัญกับคุณ แต่ไม่ได้แคร์กับเรื่องที่ไม่จำเป็น คุณรู้ว่าตัวเองต้องการอะไรและจัดการได้ในแบบของตัวเอง"),
        ("กลางๆ เอียงอิสระ",
         "คุณวางแผนได้เมื่อจำเป็น แต่ไม่ได้ชอบถูกผูกไว้กับตารางเป๊ะๆ มีระเบียบเพียงพอที่จะทำสิ่งต่างๆ ได้ แต่ก็มีพื้นที่สำหรับ spontaneity เสมอ"),
        ("สมดุลดี",
         "คุณมีทั้งระเบียบและความยืดหยุ่นในสัดส่วนที่พอดี วางแผนได้เมื่อต้องการ ปรับเปลี่ยนได้เมื่อจำเป็น"),
        ("กลางๆ เอียงมีระเบียบ",
         "คุณชอบมีโครงสร้างพอสมควร และมักทำตามแผนที่วางไว้ได้ดีกว่าคนส่วนใหญ่ รู้สึกดีกว่าเมื่อทุกอย่างอยู่ในที่ของมัน"),
        ("ค่อนข้างมีระเบียบ",
         "คุณรับผิดชอบได้ดีและมักทำงานเสร็จก่อนกำหนด ความเป็นระเบียบของคุณทำให้คนอื่นไว้วางใจได้"),
        ("วางแผนเก่ง",
         "คุณมีระบบชัดเจนในการจัดการชีวิต ไม่ว่าจะเป็น to-do list, ตาราง, หรือแผนระยะยาว คนรอบข้างมักพึ่งพาคุณในเรื่องที่ต้องการความรอบคอบ"),
        ("มีระเบียบมากๆ",
         "คุณทำงานอย่างรอบคอบและตั้งใจในทุกอย่างที่ทำ ชีวิตคุณมีโครงสร้างที่ชัดเจน และคุณรู้สึกดีที่สุดเมื่อทุกอย่างอยู่ในที่ที่ควรอยู่"),
    ],
    "OPN": [
        ("ชอบสิ่งคุ้นเคย",
         "คุณให้ความสำคัญกับสิ่งที่ได้ผลจริงๆ มากกว่าการทดลองใหม่ๆ ความมั่นคงและสิ่งที่รู้จักดีมีคุณค่ามากสำหรับคุณ"),
        ("ปฏิบัตินิยม",
         "คุณมองโลกในแง่ที่จับต้องได้ ไม่ค่อยหลงไปกับทฤษฎีหรือจินตนาการ คุณเน้นสิ่งที่ได้ผลจริงในชีวิตจริง"),
        ("ค่อนข้างปฏิบัติจริง",
         "คุณเปิดรับสิ่งใหม่บ้างเมื่อมันพิสูจน์ตัวเองแล้ว ไม่ได้รีบตะครุบทุกเทรนด์ใหม่ แต่ก็ไม่ได้ปิดกั้นตัวเองจากการเปลี่ยนแปลง"),
        ("กลางๆ เอียงปฏิบัติ",
         "คุณมีความคิดสร้างสรรค์เป็นระยะๆ แต่มักกลับมาสู่สิ่งที่รู้จักและเชื่อถือได้ สมดุลระหว่างการสำรวจและความปลอดภัย"),
        ("กลางๆ",
         "คุณสนใจทั้งสิ่งใหม่และสิ่งที่คุ้นเคย บางวันอยากลองอะไรแปลกๆ บางวันก็ขอสิ่งเดิมที่ไว้ใจได้"),
        ("กลางๆ เอียงสำรวจ",
         "คุณชอบแนวคิดใหม่ๆ และมักพบว่าตัวเองถูกดึงดูดไปยังหัวข้อที่หลากหลาย ความอยากรู้ของคุณนำคุณไปในที่ที่น่าสนใจเสมอ"),
        ("ชอบสำรวจ",
         "คุณมีจินตนาการที่สดใสและชอบคิดนอกกรอบ แนวคิดซับซ้อนและคำถามปรัชญาทำให้คุณสนุก ไม่ใช่ปวดหัว"),
        ("นักจินตนาการ",
         "โลกในหัวคุณมีความสมบูรณ์และน่าสนใจมาก คุณเชื่อมโยงแนวคิดที่คนอื่นไม่เห็นความสัมพันธ์ได้ ความคิดสร้างสรรค์ของคุณเป็นทรัพยากรที่ทรงพลัง"),
        ("นักฝันสุดขีด",
         "จินตนาการคุณทำงานตลอดเวลา คุณเห็นความเป็นไปได้ในทุกที่และคิดในเชิงนามธรรมได้เป็นธรรมชาติ โลกต้องการคนที่ฝันแบบคุณ"),
    ],
}

# Highlight paragraphs — 9 slots, same slot_idx logic
HIGHLIGHT_PARAGRAPHS: dict[str, list[str]] = {
    "CSN6": [
        "แทบไม่เคยเลย ของอยู่ที่ไหนคุณรู้หมด ความจำเรื่องของใช้แม่นยำมาก",
        "บางครั้งลืมบ้าง แต่หาเจอได้ไม่นาน ไม่ได้เป็นปัญหาในชีวิตประจำวัน",
        "ลืมเป็นระยะๆ แต่ไม่ได้รบกวนชีวิตมาก ส่วนใหญ่หาเจอในที่สุด",
        "ลืมค่อนข้างบ่อย บางทีหาของนานพอสมควร แต่ก็ยังพอรับได้",
        "วางของแล้วจำไม่ได้บ่อยพอสมควร เริ่มรู้สึกว่ามันรบกวนบ้างเป็นบางครั้ง",
        "ลืมบ่อยพอที่จะสังเกตได้ชัด หาของนานเป็นเรื่องปกติในชีวิตประจำวัน",
        "วางของแล้วลืมค่อนข้างบ่อย เป็นส่วนหนึ่งของชีวิตไปแล้ว",
        "ลืมบ่อยมาก หาของเป็นกิจวัตรประจำวันที่ทำเป็นประจำ",
        "วางของแล้วจำไม่ได้บ่อยมากจนเกือบทุกวัน มันกลายเป็นเรื่องปกติในชีวิตไปแล้ว",
    ],
    "AGR9": [
        "ไม่ค่อยจับได้ว่าคนอื่นรู้สึกยังไง ต้องมีสัญญาณชัดมากๆ ถึงจะรู้",
        "จับได้บ้างแต่ต้องมีสัญญาณชัดๆ ไม่ได้อ่านออกโดยอัตโนมัติ",
        "จับได้พอสมควรในสถานการณ์ที่คุ้นเคย กับคนไม่รู้จักก็ยังต้องเดาอยู่",
        "อ่านอารมณ์ได้ค่อนข้างดีในหลายสถานการณ์ แต่ก็ยังผิดบ้างเป็นบางครั้ง",
        "อ่านอารมณ์คนได้พอสมควร มักรู้สึกได้ว่าบรรยากาศในห้องเป็นยังไง",
        "อ่านอารมณ์คนได้ค่อนข้างแม่น มักรู้ก่อนที่เขาจะพูดออกมา",
        "อ่านอารมณ์คนได้ดีมาก เหมือนมีเซ็นเซอร์พิเศษ รู้สึกได้แทบจะทันที",
        "อ่านอารมณ์คนได้แม่นมาก ทั้งจากสีหน้า น้ำเสียง และภาษากาย",
        "อ่านอารมณ์คนได้แม่นมากเกือบทุกครั้ง เหมือนมีเรดาร์พิเศษในการรับรู้ความรู้สึกคนอื่น",
    ],
    "AGR5": [
        "แคร์มากๆ เลย ปัญหาคนรอบข้างคือเรื่องของเราด้วยเสมอ",
        "แคร์พอสมควร มักอยากช่วยเมื่อมีโอกาส",
        "แคร์บ้างขึ้นอยู่กับว่าสนิทแค่ไหน ไม่ได้เป็นยังงั้นกับทุกคน",
        "กลางๆ บางเรื่องก็แคร์ บางเรื่องก็ปล่อย ขึ้นอยู่กับสถานการณ์",
        "ส่วนใหญ่ปล่อยให้เป็นเรื่องของเขา แต่ก็ไม่ได้เย็นชาซะทีเดียว",
        "มักปล่อยให้เป็นเรื่องของคนอื่น ไม่ได้รู้สึกว่าต้องยื่นมือเข้าไปช่วย",
        "ปัญหาของคนอื่นไม่ค่อยเกี่ยวกับเราเท่าไหร่ แต่ก็ไม่ได้แข็งใจซะทีเดียว",
        "ไม่ค่อยรู้สึกว่าต้องแบกปัญหาของคนอื่น ทุกคนต้องจัดการตัวเองได้",
        "ปัญหาของคนอื่นเป็นเรื่องของเขา ไม่ใช่หน้าที่ของเราที่จะต้องแบกไว้",
    ],
    "EXT4": [
        "ไม่เลยเลย ชอบอยู่แถวหน้า เป็นจุดสนใจเป็นเรื่องปกติ",
        "ส่วนใหญ่ชอบโดดเด่น แต่ก็ถอยได้บ้างเมื่อจำเป็น",
        "กลางๆ บางทีก็ชอบ spotlight บางทีก็โอเคที่จะเงียบ",
        "เอียงไปทางชอบอยู่แถวหลังมากกว่า แต่ก็ไม่ได้หนีเสมอไป",
        "ชอบอยู่แถวหลังพอสมควร ปล่อยให้คนอื่น shine ดีกว่า",
        "ค่อนข้างชอบอยู่แถวหลัง ไม่ชอบให้ทุกสายตามาจับจ้อง",
        "ชอบหลบอยู่เงียบๆ ดีกว่า การเป็นจุดสนใจทำให้อึดอัดนิดนึง",
        "ชอบอยู่แถวหลังมาก ขอสังเกตการณ์อยู่ห่างๆ ดีกว่า",
        "ไม่ชอบเป็นจุดสนใจเลย ขอหลบอยู่หลังๆ เงียบๆ ดีที่สุด",
    ],
    "EST7": [
        "นิ่งมากๆ อารมณ์แทบไม่กระเทือนง่ายๆ คงเส้นคงวาสูง",
        "ค่อนข้างนิ่ง อารมณ์เปลี่ยนบ้างแต่ไม่บ่อย และไม่ได้รุนแรง",
        "บางครั้งก็แปรปรวนบ้างตามเหตุการณ์ แต่โดยรวมยังคุมได้ดี",
        "เปลี่ยนเป็นระยะๆ แต่ยังพอคาดเดาได้ ไม่ได้กระทบชีวิตมาก",
        "อารมณ์เปลี่ยนพอสมควร บางวันดี บางวันไม่ดี โดยไม่ได้มีเหตุชัดเจน",
        "เปลี่ยนค่อนข้างบ่อย คนใกล้ชิดสังเกตได้ บางทีตัวเองก็งงว่าทำไม",
        "อารมณ์แปรปรวนบ่อย ขึ้นๆ ลงๆ ได้ในวันเดียว",
        "อารมณ์เปลี่ยนเร็วและบ่อยมาก บางทีตัวเองก็ตามแทบไม่ทัน",
        "อารมณ์แปรปรวนมากๆ เปลี่ยนได้ทันทีโดยไม่รู้ตัว เป็นส่วนหนึ่งของชีวิตประจำวัน",
    ],
    "OPN3": [
        "ไม่ค่อยมีเลย คิดในแง่ปฏิบัติมากกว่า ภาพในหัวไม่ค่อยชัด",
        "มีบ้างแต่ไม่ได้คิดฝันบ่อยๆ จินตนาการมาเป็นครั้งคราว",
        "มีจินตนาการพอควร ไม่มากไม่น้อย บางทีก็มีภาพในหัว",
        "จินตนาการค่อนข้างดี มีภาพในหัวได้เป็นระยะๆ",
        "จินตนาการพอสมควร มีโลกส่วนตัวที่ทำงานบ้างเป็นบางเวลา",
        "จินตนาการค่อนข้างดี ภาพในหัวชัดพอสมควร มีอยู่บ่อยๆ",
        "จินตนาการดีและชัดเจน มีภาพและเรื่องราวในหัวได้ง่าย",
        "จินตนาการชัดและสมบูรณ์มาก มีโลกส่วนตัวที่ทำงานตลอดเวลา",
        "จินตนาการทำงานตลอดเวลา โลกในหัวสมบูรณ์มาก ภาพชัดเสมือนจริง",
    ],
    "OPN10": [
        "แทบไม่มีเลย ชอบทำสิ่งที่คุ้นเคยมากกว่าคิดเรื่องใหม่ๆ",
        "มีบ้างแต่ไม่บ่อย ไอเดียใหม่มาเป็นครั้งคราวเท่านั้น",
        "มีไอเดียใหม่เป็นระยะๆ ไม่ได้มาตลอดแต่ก็ไม่ได้หายาก",
        "มีไอเดียมาพอสมควร บางวันก็คิดได้เยอะกว่าปกติ",
        "ไอเดียมาสม่ำเสมอพอสมควร ไม่ได้หยุดคิดนานๆ",
        "ไอเดียมาค่อนข้างบ่อย บางทีหยุดไม่ค่อยได้",
        "ไอเดียมาบ่อยมาก ต้องหาที่จดตลอด กลัวลืม",
        "ไอเดียไม่หยุด คิดตลอดเวลา บางทีล้นหัวจนจัดการไม่ทัน",
        "ไอเดียพุ่งพล่านตลอดเวลา หยุดคิดไม่ได้เลย สมองทำงานไม่หยุด",
    ],
}

# Per-trait MAE on 1–5 scale vs number of answered items K (from offline eval).
_MAE_K_ANCHORS: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 43, 49)
_MAE_ROWS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.2480, 0.4574, 0.2493, 0.2758, 0.2621),
    (0.2366, 0.3944, 0.2318, 0.2573, 0.2387),
    (0.2216, 0.3370, 0.2137, 0.2377, 0.2154),
    (0.2038, 0.2849, 0.1946, 0.2166, 0.1925),
    (0.1834, 0.2380, 0.1742, 0.1939, 0.1694),
    (0.1607, 0.1948, 0.1522, 0.1695, 0.1456),
    (0.1348, 0.1544, 0.1277, 0.1423, 0.1203),
    (0.1042, 0.1137, 0.0987, 0.1102, 0.0917),
    (0.0816, 0.0869, 0.0773, 0.0864, 0.0712),
    (0.0156, 0.0160, 0.0148, 0.0166, 0.0135),
)


def _mean_trait_mae_at_k(k: float) -> float:
    if k >= 50:
        return 0.0
    if k <= 0:
        return sum(_MAE_ROWS[0]) / 5.0
    anchors = _MAE_K_ANCHORS
    rows = _MAE_ROWS
    if k <= anchors[0]:
        return sum(rows[0]) / 5.0
    if k >= anchors[-1]:
        return sum(rows[-1]) / 5.0
    for i in range(len(anchors) - 1):
        lo, hi = anchors[i], anchors[i + 1]
        if lo <= k <= hi:
            t = (k - lo) / (hi - lo) if hi != lo else 0.0
            blended = tuple(
                rows[i][j] * (1 - t) + rows[i + 1][j] * t for j in range(5)
            )
            return sum(blended) / 5.0
    return sum(rows[-1]) / 5.0


def trust_pct_from_answered_count(n_answered: int) -> int:
    if n_answered >= 50:
        return 100
    if n_answered <= 0:
        return 5
    mm = _mean_trait_mae_at_k(float(n_answered))
    raw = 100.0 - mm * 60.0
    return max(5, min(100, round(raw)))


def slot_idx(score: float) -> int:
    """Map a 1–5 score to one of 9 paragraph slots (each 0.5 wide)."""
    return min(8, int((score - 1) / 0.5))


def score_label(s: float) -> str:
    if s < 2:
        return "ต่ำมาก"
    if s < 2.75:
        return "ต่ำ"
    if s < 3.5:
        return "ปานกลาง"
    if s < 4.25:
        return "สูง"
    return "สูงมาก"


def inject_css() -> None:
    st.markdown(
        """
<style>
    :root, .stApp, [data-testid="stAppViewContainer"], section.main {
        color-scheme: dark !important;
    }
    .stApp {
        background-color: #0a0910 !important;
        color: #f0ecff !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0a0910 !important;
    }
    section.main {
        background-color: transparent !important;
    }
    .block-container { padding-top: 2rem !important; max-width: 640px !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 1.5rem; }
    .stat-box {
        background: #13101e; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        padding: 16px; text-align: center;
    }
    .stat-num { font-size: 1.9rem; color: #f0ecff; }
    .stat-lbl { font-size: 11px; color: #8a84a3; letter-spacing: 0.04em; margin-top: 4px; }
    .trait-pill {
        display: inline-block; font-size: 12px; padding: 6px 14px; border-radius: 99px;
        border: 1px solid; margin: 4px 4px 4px 0;
    }
    .quiz-card {
        background: #13101e; border: 1px solid rgba(255,255,255,0.14); border-radius: 22px;
        padding: 24px; margin: 16px 0; border-top: 3px solid var(--accent-trait);
    }
    .meme-card, .sec-card {
        background: #13101e; border: 1px solid rgba(255,255,255,0.08); border-radius: 18px;
        padding: 18px; margin-bottom: 12px;
    }
    .bar-track { height: 8px; background: rgba(255,255,255,0.07); border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
    .eyebrow { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: #8a84a3; margin-bottom: 1rem; }
    h1 { font-weight: 400 !important; color: #f0ecff !important; }
    .muted { color: #8a84a3; font-size: 15px; line-height: 1.75; }
    .section-head { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #8a84a3; margin: 2rem 0 1rem; }

    /* ── Paragraph result cards ── */
    .trait-card {
        background: #13101e;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 3px solid var(--tc);
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 10px;
    }
    .trait-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 10px;
        gap: 12px;
    }
    .trait-card-name { font-size: 15px; font-weight: 500; color: #f0ecff; }
    .trait-card-th { font-size: 11px; color: #8a84a3; margin-top: 2px; }
    .trait-card-tag {
        font-size: 12px;
        font-weight: 500;
        color: var(--tc);
        padding: 4px 12px;
        border-radius: 99px;
        border: 1px solid var(--tc);
        white-space: nowrap;
        opacity: 0.9;
        flex-shrink: 0;
    }
    .trait-card-para {
        font-size: 14px;
        color: #a09bbf;
        line-height: 1.75;
        margin: 0;
    }

    /* ── Highlight paragraph cards ── */
    .hl-card {
        background: #13101e;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .hl-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        gap: 10px;
    }
    .hl-card-label { font-size: 14px; font-weight: 500; color: #f0ecff; }
    .hl-answered-tag {
        font-size: 10px; padding: 3px 10px; border-radius: 99px;
        color: #b9f5e8; border: 1px solid rgba(100,220,200,0.4);
        background: rgba(30,80,70,0.3); flex-shrink: 0;
    }
    .hl-card-para {
        font-size: 13px;
        color: #a09bbf;
        line-height: 1.65;
        margin: 0;
    }

    button.b5-restart-sticky-btn {
        background-color: #13101e !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        color: #8a84a3 !important;
    }
    button.b5-restart-sticky-btn:hover {
        border-color: rgba(255, 100, 100, 0.45) !important;
        color: #ff9a9a !important;
    }
    .stApp button[data-testid="baseButton-primary"],
    .stApp button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, rgba(127, 119, 221, 0.38), rgba(83, 74, 183, 0.22)) !important;
        border: 1px solid rgba(127, 119, 221, 0.48) !important;
        color: #ece9ff !important;
    }
    .stApp button[data-testid="baseButton-primary"]:hover:not(:disabled),
    .stApp button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        background: rgba(127, 119, 221, 0.42) !important;
        border-color: rgba(160, 150, 240, 0.65) !important;
        color: #ffffff !important;
    }
    .stApp button[data-testid="baseButton-primary"]:disabled,
    .stApp button[data-testid="stBaseButton-primary"]:disabled {
        background: rgba(127, 119, 221, 0.12) !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        color: #6a6288 !important;
    }
    .stApp button[data-testid="baseButton-secondary"],
    .stApp button[data-testid="stBaseButton-secondary"] {
        background: #1c1828 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #d4cfe8 !important;
    }
    .stApp button[data-testid="baseButton-secondary"]:hover:not(:disabled),
    .stApp button[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
        border-color: rgba(255, 255, 255, 0.18) !important;
        background: rgba(127, 119, 221, 0.1) !important;
    }
    .stApp button[data-testid="baseButton-secondary"]:disabled,
    .stApp button[data-testid="stBaseButton-secondary"]:disabled {
        opacity: 0.45 !important;
    }
    [data-testid="stProgress"] > div { background-color: rgba(255,255,255,0.12) !important; }
    div[data-testid="stCaption"] { color: #8a84a3 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def pin_result_sticky_footer() -> None:
    html = r"""
<script>
(function () {
  function pin() {
    try {
      var doc = window.parent.document;
      if (!doc.querySelector(".results-footer-marker")) return;
      var btns = doc.querySelectorAll("button");
      var target = null;
      for (var i = 0; i < btns.length; i++) {
        var tx = (btns[i].innerText || "").trim();
        if (tx.indexOf("เล่นใหม่อีกรอบ") !== -1) { target = btns[i]; break; }
      }
      if (!target) return;
      target.classList.add("b5-restart-sticky-btn");
      var el = target;
      var footer = null;
      while (el && el.parentElement) {
        if (el.parentElement.getAttribute("data-testid") === "stVerticalBlock") {
          footer = el;
          break;
        }
        el = el.parentElement;
      }
      if (!footer) return;
      footer.style.setProperty("position", "fixed", "important");
      footer.style.setProperty("bottom", "0", "important");
      footer.style.setProperty("left", "0", "important");
      footer.style.setProperty("right", "0", "important");
      footer.style.setProperty("z-index", "999999", "important");
      footer.style.setProperty("display", "flex", "important");
      footer.style.setProperty("justify-content", "center", "important");
      footer.style.setProperty(
        "padding",
        "16px 20px max(16px, env(safe-area-inset-bottom, 0px))",
        "important"
      );
      footer.style.setProperty("margin", "0", "important");
      footer.style.setProperty(
        "background",
        "linear-gradient(to top, #0a0910 60%, transparent)",
        "important"
      );
      footer.style.setProperty("width", "100%", "important");
      footer.style.setProperty("box-sizing", "border-box", "important");
      var inner = footer.firstElementChild;
      if (inner) {
        inner.style.setProperty("width", "100%", "important");
        inner.style.setProperty("max-width", "640px", "important");
      }
      var main = doc.querySelector('[data-testid="stMain"]') || doc.querySelector("section.main");
      if (main) {
        var bc = main.querySelector(".block-container");
        if (bc) bc.style.setProperty("padding-bottom", "5.75rem", "important");
      }
    } catch (e) {}
  }
  pin();
  setTimeout(pin, 0);
  setTimeout(pin, 80);
  setTimeout(pin, 250);
})();
</script>
"""
    components.html(html, height=1)


@st.cache_resource
def load_predictor() -> dict[str, Any] | None:
    try:
        import joblib
    except ImportError:
        return None
    if not ARTIFACT_PATH.is_file():
        return None

    try:
        bundle = joblib.load(ARTIFACT_PATH)
    except Exception:
        return None
    ridge = bundle["ridge"]
    all_q_cols: list[str] = list(bundle["all_q_cols"])
    trait_cols: dict[str, list[str]] = dict(bundle["trait_cols"])
    meme_questions: list[str] = list(bundle["meme_questions"])

    col_to_idx = {c: i for i, c in enumerate(all_q_cols)}
    trait_idxs = {
        trait: np.array([col_to_idx[c] for c in qs], dtype=np.int64)
        for trait, qs in trait_cols.items()
    }

    def make_features(x_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.hstack([x_masked, mask]).astype(np.float32)

    def predict_ridge(x_masked: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pred = ridge.predict(make_features(x_masked, mask))
        pred = np.where(mask == 1, x_masked, pred)
        return np.clip(pred, 1, 5)

    def trait_scores_from_answers(x: np.ndarray) -> np.ndarray:
        out = np.empty((len(x), len(trait_idxs)), dtype=np.float32)
        for ti, idxs in enumerate(trait_idxs.values()):
            out[:, ti] = x[:, idxs].mean(axis=1)
        return out

    def predict_personality(answers: dict[str, float]) -> dict[str, Any]:
        x = np.zeros((1, len(all_q_cols)), dtype=np.float32)
        m = np.zeros((1, len(all_q_cols)), dtype=np.float32)
        for q, v in answers.items():
            if q in col_to_idx:
                x[0, col_to_idx[q]] = float(v)
                m[0, col_to_idx[q]] = 1.0

        pred = predict_ridge(x, m)[0]
        traits = trait_scores_from_answers(pred[None, :])[0]
        answered_set = set(answers)

        def split(q_codes: list[str]) -> dict[str, dict[str, float]]:
            ans: dict[str, float] = {}
            prd: dict[str, float] = {}
            for q in q_codes:
                val = round(float(pred[col_to_idx[q]]), 2)
                (ans if q in answered_set else prd)[q] = val
            return {"answered": ans, "predicted": prd}

        secondary_qs = [q for q in all_q_cols if q not in meme_questions]

        return {
            "n_answered": int(m.sum()),
            "trait_scores": dict(zip(trait_cols.keys(), traits.round(3).tolist())),
            "primary": split(meme_questions),
            "secondary": split(secondary_qs),
        }

    return {
        "predict_personality": predict_personality,
        "meme_questions": meme_questions,
        "all_q_cols": all_q_cols,
    }


def init_session() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    if "quiz_order" not in st.session_state:
        st.session_state.quiz_order = []
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "q_index" not in st.session_state:
        st.session_state.q_index = 0


def reset_all() -> None:
    st.session_state.page = "welcome"
    st.session_state.quiz_order = []
    st.session_state.answers = {}
    st.session_state.q_index = 0
    for k in list(st.session_state.keys()):
        if str(k).startswith("pick_") or str(k).startswith("rate_"):
            del st.session_state[k]


def start_quiz(pred_bundle: dict[str, Any] | None) -> None:
    import random

    raw_meme = (
        tuple(pred_bundle["meme_questions"]) if pred_bundle else MEME_QIDS_FALLBACK
    )
    meme = [q for q in raw_meme if q in ALL_QUESTIONS]
    if not meme:
        meme = [q for q in MEME_QIDS_FALLBACK if q in ALL_QUESTIONS]
    meme_set = set(meme)
    rest = [q for q in ALL_QUESTIONS if q not in meme_set]
    random.shuffle(rest)
    tail = list(meme)
    random.shuffle(tail)
    order = rest + tail

    st.session_state.quiz_order = order
    st.session_state.answers = {}
    st.session_state.q_index = 0
    st.session_state.page = "quiz"


def render_welcome(pred_bundle: dict[str, Any] | None) -> None:
    st.markdown('<p class="eyebrow">Big Five Personality</p>', unsafe_allow_html=True)
    st.markdown("## คุณเป็น _คนแบบไหน_ กันแน่?")
    st.markdown(
        '<p class="muted">แบบทดสอบบุคลิกภาพ 50 ข้อ เทรนมาจากคนกว่า 1 ล้านคน '
        "— ตอบแค่ 5 ข้อก็ดูผลได้เลย ที่เหลือ AI จะดูให้เอง 🧠</p>",
        unsafe_allow_html=True,
    )

    pills = ""
    for tid, spec in TRAITS.items():
        c = TRAIT_COLORS[tid]
        pills += (
            f'<span class="trait-pill" style="color:{c};border-color:{c}55;'
            f'background:{c}18">{spec["name"]} · {spec["th"]}</span> '
        )
    st.markdown(pills, unsafe_allow_html=True)

    st.markdown(
        """
<div class="stat-grid">
  <div class="stat-box"><div class="stat-num">50</div><div class="stat-lbl">จำนวณข้อทั้งหมด</div></div>
  <div class="stat-box"><div class="stat-num">5+</div><div class="stat-lbl">ตอบแค่นี้ก็พอ</div></div>
  <div class="stat-box"><div class="stat-num">AI</div><div class="stat-lbl">ยิ่งตอบมาก AI ยิ่งทายผลแม่น</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("ลองเลย →", type="primary", use_container_width=True):
        start_quiz(pred_bundle)
        st.rerun()


def render_quiz(pred_bundle: dict[str, Any] | None) -> None:
    if pred_bundle is None:
        st.error(
            f"ไม่พบโมเดลที่ `{ARTIFACT_PATH}`. รันเซลล์บันทึกโมเดลใน `models/main.ipynb` ก่อน"
        )
        return

    st.markdown(
        '<span class="b5-quiz-page" aria-hidden="true" style="display:none"></span>',
        unsafe_allow_html=True,
    )

    order = st.session_state.quiz_order
    idx = st.session_state.q_index
    answers: dict[str, int] = st.session_state.answers
    n_ans = len(answers)
    total = len(order)
    qid = order[idx]
    trait, qtext = ALL_QUESTIONS[qid]
    trait_color = TRAIT_COLORS[trait]

    st.caption(f"ข้อ **{idx + 1}** / **{total}**")
    st.progress((idx + 1) / total)
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"ตอบไปแล้ว **{n_ans}** ข้อ")
    with c2:
        st.caption(f"~**{trust_pct_from_answered_count(n_ans)}**% แม่น")

    st.markdown(
        f'<div class="quiz-card" style="--accent-trait: {trait_color}">'
        f'<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{trait_color};margin-bottom:12px;font-weight:500">'
        f'{TRAITS[trait]["name"]} — {TRAITS[trait]["th"]}</div>'
        f'<div style="font-size:1.45rem;line-height:1.45;color:#f0ecff">{qtext}</div></div>',
        unsafe_allow_html=True,
    )

    pick_key = f"pick_{qid}"
    if pick_key not in st.session_state:
        st.session_state[pick_key] = answers.get(qid)

    cols = st.columns(5)
    for val in range(1, 6):
        with cols[val - 1]:
            picked = st.session_state[pick_key] == val
            if st.button(
                f"{val}\n{RATING_CAPS[val - 1]}",
                key=f"b_{qid}_{val}",
                use_container_width=True,
                type="primary" if picked else "secondary",
            ):
                st.session_state[pick_key] = val
                st.rerun()

    selected = st.session_state[pick_key]
    can_next = selected is not None

    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button(
            "← กลับ",
            use_container_width=True,
            type="secondary",
            disabled=idx <= 0,
            key=f"quiz_back_{qid}",
        ):
            if idx > 0:
                st.session_state.q_index -= 1
                st.rerun()
    with b2:
        next_label = "ดูผลลัพธ์ →" if idx >= total - 1 else "บันทึกและถัดไป →"
        if st.button(
            next_label,
            type="primary",
            use_container_width=True,
            disabled=not can_next,
            key=f"quiz_next_{qid}",
        ):
            if selected is not None:
                answers[qid] = int(selected)
                st.session_state.answers = answers
                if idx + 1 < total:
                    st.session_state.q_index += 1
                else:
                    st.session_state.page = "result"
                st.rerun()

    if n_ans >= 5:
        st.markdown("---")
        if st.button(
            "✦ พอแล้ว ดูผลได้เลย! — แต่ยิ่งตอบมาก AI ยิ่งทายผลเเม่นขึ้นนะ ✨",
            use_container_width=True,
            type="secondary",
            key="quiz_early_finish",
        ):
            st.session_state.page = "result"
            st.rerun()


# ---------------------------------------------------------------------------
# New summary result page — paragraph-based, no bars
# ---------------------------------------------------------------------------

def render_results(pred_bundle: dict[str, Any] | None) -> None:
    if pred_bundle is None:
        st.error("ไม่พบโมเดล — ไม่สามารถแสดงผลได้")
        return

    st.markdown(
        '<div class="results-footer-marker" aria-hidden="true" style="display:none"></div>',
        unsafe_allow_html=True,
    )

    answers = {k: float(v) for k, v in st.session_state.answers.items()}
    predict_fn: Callable = pred_bundle["predict_personality"]
    meme_order: list[str] = pred_bundle["meme_questions"]

    out = predict_fn(answers)
    n_answered = out["n_answered"]
    n_fill = 50 - n_answered
    conf = trust_pct_from_answered_count(n_answered)
    trait_scores: dict[str, float] = out["trait_scores"]
    primary = out["primary"]

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<p class="eyebrow">ผลลัพธ์ของคุณ</p>', unsafe_allow_html=True)
    st.markdown("### นี่คือตัวตนของคุณ ✦")
    st.markdown(
        f'<p class="muted">เราประกอบภาพบุคลิกครบ 50 หัวข้อ — {n_answered} ข้อจากคำตอบของคุณ '
        f"และ {n_fill} ข้อจากโมเดลเติม</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="stat-grid" style="grid-template-columns:1fr 1fr 1fr">
  <div class="stat-box"><div class="stat-num">{n_answered}<small style="opacity:0.6">/50</small></div>
  <div class="stat-lbl">คำตอบของคุณ</div></div>
  <div class="stat-box"><div class="stat-num">{n_fill}</div><div class="stat-lbl">โมเดลเติมให้</div></div>
  <div class="stat-box"><div class="stat-num">{conf}<small style="opacity:0.6">%</small></div>
  <div class="stat-lbl">ความมั่นใจ</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Big Five paragraph cards ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Big Five — ภาพรวม 5 ด้าน</div>', unsafe_allow_html=True)

    trait_order = ["EXT", "EST", "AGR", "CSN", "OPN"]
    for tid in trait_order:
        if tid not in trait_scores:
            continue
        score = float(trait_scores[tid])
        color = TRAIT_COLORS[tid]
        spec = TRAITS[tid]
        si = slot_idx(score)
        tag, para = TRAIT_PARAGRAPHS[tid][si]

        st.markdown(
            f'<div class="trait-card" style="--tc:{color}">'
            f'<div class="trait-card-header">'
            f'<div><div class="trait-card-name">{spec["name"]}</div>'
            f'<div class="trait-card-th">{spec["th"]}</div></div>'
            f'<span class="trait-card-tag">{tag}</span>'
            f'</div>'
            f'<p class="trait-card-para">{para}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Highlight paragraph cards ────────────────────────────────────────────
    st.markdown('<div class="section-head">ไฮไลต์สำคัญ 7 อย่าง</div>', unsafe_allow_html=True)

    for mid in meme_order:
        if mid not in HIGHLIGHT_PARAGRAPHS or mid not in ALL_QUESTIONS:
            continue

        tr = ALL_QUESTIONS[mid][0]
        color = TRAIT_COLORS[tr]
        meta = MEME_META.get(mid, {})
        label = meta.get("label", mid)
        val = float(primary["answered"].get(mid, primary["predicted"].get(mid, 3.0)))
        is_user = mid in answers
        si = slot_idx(val)
        para = HIGHLIGHT_PARAGRAPHS[mid][si]

        answered_tag = (
            '<span class="hl-answered-tag">คุณตอบ</span>' if is_user else ""
        )
        st.markdown(
            f'<div class="hl-card">'
            f'<div class="hl-card-header">'
            f'<span class="hl-card-label" style="color:{color}">{label}</span>'
            f'{answered_tag}'
            f'</div>'
            f'<p class="hl-card-para">{para}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Deep-dive button → detail page ───────────────────────────────────────
    st.markdown(
        """
<div style="font-size:12px;color:#5a5472;line-height:1.7;padding:18px;background:#13101e;
border:1px solid rgba(255,255,255,0.08);border-radius:12px;margin-top:24px;margin-bottom:8px">
<strong style="color:#8a84a3">หมายเหตุ:</strong> ค่าที่ AI เติมเป็นการประมาณทางสถิติ ไม่ใช่คำวินิจฉัยทางจิตวิทยานะ ·
สเกล 1–5 ตามมาตรฐาน IPIP · ข้อมูลจาก Big Five Personality Test (Kaggle, Nov 2018)
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "ดูตัวเลขและ AI เดาแบบละเอียด →",
        use_container_width=True,
        type="secondary",
        key="go_to_detail",
    ):
        st.session_state.page = "result_detail"
        st.rerun()

    if st.button(
        "← เล่นใหม่อีกรอบ",
        use_container_width=True,
        type="secondary",
        key="result_restart_open_dialog",
    ):
        restart_dialog()

    pin_result_sticky_footer()


# ---------------------------------------------------------------------------
# Original detailed result page (numbers + AI predictions)
# ---------------------------------------------------------------------------

def render_results_detail(pred_bundle: dict[str, Any] | None) -> None:
    if pred_bundle is None:
        st.error("ไม่พบโมเดล — ไม่สามารถแสดงผลได้")
        return

    st.markdown(
        '<div class="results-footer-marker" aria-hidden="true" style="display:none"></div>',
        unsafe_allow_html=True,
    )

    answers = {k: float(v) for k, v in st.session_state.answers.items()}
    predict_fn: Callable = pred_bundle["predict_personality"]
    meme_order: list[str] = pred_bundle["meme_questions"]

    out = predict_fn(answers)
    n_answered = out["n_answered"]
    n_fill = 50 - n_answered
    conf = trust_pct_from_answered_count(n_answered)
    trait_scores: dict[str, float] = out["trait_scores"]
    primary = out["primary"]
    secondary = out["secondary"]

    st.markdown('<p class="eyebrow">รายละเอียดเชิงลึก</p>', unsafe_allow_html=True)
    st.markdown("### ตัวเลขและ AI เดา 👀")
    st.markdown(
        f'<p class="muted">เราประกอบภาพบุคลิกครบ 50 หัวข้อ — {n_answered} ข้อจากคำตอบของคุณ '
        f"และ {n_fill} ข้อจากโมเดลเติม</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="stat-grid" style="grid-template-columns:1fr 1fr 1fr">
  <div class="stat-box"><div class="stat-num">{n_answered}<small style="opacity:0.6">/50</small></div>
  <div class="stat-lbl">คำตอบของคุณ</div></div>
  <div class="stat-box"><div class="stat-num">{n_fill}</div><div class="stat-lbl">โมเดลเติมให้</div></div>
  <div class="stat-box"><div class="stat-num">{conf}<small style="opacity:0.6">%</small></div>
  <div class="stat-lbl">ความมั่นใจ</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">Big Five — ภาพรวม 5 ด้าน</div>', unsafe_allow_html=True)
    trait_order = ["EXT", "EST", "AGR", "CSN", "OPN"]
    for tid in trait_order:
        if tid not in trait_scores:
            continue
        spec = TRAITS[tid]
        trait_bar_row(tid, spec["name"], spec["th"], float(trait_scores[tid]))

    st.markdown('<div class="section-head">ไฮไลต์สำคัญ 7 อย่าง</div>', unsafe_allow_html=True)
    for mid in meme_order:
        if mid not in MEME_META or mid not in ALL_QUESTIONS:
            continue
        meta = MEME_META[mid]
        tr = ALL_QUESTIONS[mid][0]
        val = float(primary["answered"].get(mid, primary["predicted"].get(mid, 3.0)))
        is_user = mid in answers
        pct = (val - 1) / 4 * 100
        lbl = score_label(val)
        color = TRAIT_COLORS[tr]
        tag = (
            '<span style="font-size:10px;padding:3px 10px;border-radius:99px;'
            'color:#b9f5e8;border:1px solid rgba(100,220,200,0.4);background:rgba(30,80,70,0.3)">คุณตอบ</span>'
            if is_user
            else ""
        )
        st.markdown(
            f'<div class="meme-card"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
            f'<div><span style="font-size:15px;font-weight:500;color:#f0ecff">{meta["label"]}</span> {tag}</div>'
            f'<div style="font-size:2rem;color:{color}">{val:.1f}</div></div>'
            f'<p style="color:#8a84a3;font-size:13px;margin:12px 0">{meta["desc"]}</p>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#5a5472">'
            f'<span>{meta["low"]}</span><span>{meta["high"]}</span></div>'
            f'<div class="bar-track" style="height:5px;margin-top:4px">'
            f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<div style="font-size:11px;font-weight:600;color:{color};text-align:right;margin-top:6px">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-head">ที่เหลือ AI เดาให้ 👀</div>', unsafe_allow_html=True)
    sec_predicted = secondary["predicted"]
    secondary_rows: list[tuple[str, float]] = []
    for sq in pred_bundle["all_q_cols"]:
        if sq in meme_order or sq in answers:
            continue
        if sq in sec_predicted:
            secondary_rows.append((sq, float(sec_predicted[sq])))

    if not secondary_rows:
        st.info("คุณตอบครบทุกข้อรองด้วยตัวเองแล้ว ✓")
    else:
        for sq, val in secondary_rows:
            tr = ALL_QUESTIONS[sq][0]
            txt = ALL_QUESTIONS[sq][1]
            color = TRAIT_COLORS[tr]
            pct = (val - 1) / 4 * 100
            lbl = score_label(val)
            st.markdown(
                f'<div class="sec-card"><div style="display:flex;justify-content:space-between;gap:8px">'
                f'<div style="font-size:12px;color:#f0ecff;flex:1">{txt}</div>'
                f'<div style="font-size:1.25rem;color:{color}">{val:.1f}</div></div>'
                f'<div class="bar-track" style="height:4px;margin:8px 0">'
                f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'<div style="display:flex;justify-content:space-between;font-size:10px">'
                f'<span style="color:#5a5472">{tr} · {sq}</span>'
                f'<span style="font-weight:600;color:{color}">{lbl}</span></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
<div style="font-size:12px;color:#5a5472;line-height:1.7;padding:18px;background:#13101e;
border:1px solid rgba(255,255,255,0.08);border-radius:12px;margin-top:24px">
<strong style="color:#8a84a3">หมายเหตุ:</strong> ค่าที่ AI เติมเป็นการประมาณทางสถิติ ไม่ใช่คำวินิจฉัยทางจิตวิทยานะ ·
สเกล 1–5 ตามมาตรฐาน IPIP · ข้อมูลจาก Big Five Personality Test (Kaggle, Nov 2018)
</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "← กลับหน้าสรุป",
        use_container_width=True,
        type="secondary",
        key="back_to_summary",
    ):
        st.session_state.page = "result"
        st.rerun()

    if st.button(
        "← เล่นใหม่อีกรอบ",
        use_container_width=True,
        type="secondary",
        key="result_restart_open_dialog",
    ):
        restart_dialog()

    pin_result_sticky_footer()


def trait_bar_row(trait: str, name: str, th: str, score: float) -> None:
    pct = (score - 1) / 4 * 100
    color = TRAIT_COLORS[trait]
    st.markdown(
        f'<div style="display:grid;grid-template-columns:120px 1fr 52px;gap:12px;align-items:center;margin-bottom:14px">'
        f'<div><span style="font-size:13px;color:#f0ecff">{name}</span>'
        f'<span style="display:block;font-size:11px;color:#8a84a3">{th}</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<div style="font-size:1.1rem;text-align:right;color:{color}">{score:.2f}</div></div>',
        unsafe_allow_html=True,
    )


@st.dialog("เริ่มใหม่เลยเหรอ?")
def restart_dialog() -> None:
    st.markdown("คำตอบและผลลัพธ์ทั้งหมดจะหายไปนะ กดยืนยันถ้ามั่นใจแล้ว")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("ใช่เลย เริ่มใหม่!", type="primary", use_container_width=True):
            reset_all()
            st.rerun()
    with c2:
        if st.button("เดี๋ยวก่อน ดูผลต่อ", use_container_width=True):
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Big Five Personality",
        page_icon="✦",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    init_session()

    pred_bundle = load_predictor()

    if st.session_state.page == "welcome":
        render_welcome(pred_bundle)
    elif st.session_state.page == "quiz":
        render_quiz(pred_bundle)
    elif st.session_state.page == "result":
        render_results(pred_bundle)
    elif st.session_state.page == "result_detail":
        render_results_detail(pred_bundle)


if __name__ == "__main__":
    main()