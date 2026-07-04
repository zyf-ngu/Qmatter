import os
import json
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from config import  api_key

load_dotenv()
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


class SkillAgent:
    def __init__(self):
        self.skills_dir = Path("skills")
        # 第一步：加载所有技能的元数据（名称+描述，轻量级）
        self.skills_meta = self._load_skills_metadata()
        # 缓存已加载的完整技能指令
        self.loaded_skills = {}

    def _load_skills_metadata(self):
        """扫描 skills/*/SKILL.md，提取 frontmatter 中的 name 和 description"""
        meta = []
        if not self.skills_dir.exists():
            print(f"警告: 技能目录 {self.skills_dir} 不存在")
            return meta
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text(encoding='utf-8')
            # 简易解析 YAML frontmatter（```yaml ... ``` 或 --- ... ---）
            if "---" in content:
                parts = content.split("---")
                if len(parts) >= 2:
                    frontmatter = parts[1].strip()
                    name = description = ""
                    for line in frontmatter.split('\n'):
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip()
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip()
                    if name:
                        meta.append({"name": name, "description": description, "path": skill_path})
        return meta

    def _get_skills_prompt(self):
        """生成技能列表提示词（仅元数据，Token极少）"""
        if not self.skills_meta:
            return "当前没有可用技能。"
        lines = ["可用技能如下："]
        for s in self.skills_meta:
            lines.append(f"- {s['name']}: {s['description']}")
        return "\n".join(lines)

    def _select_skill_and_params(self, user_input):
        """调用 DeepSeek 判断应该使用哪个技能，并提取参数"""
        prompt = f"""你是一个智能助手。{self._get_skills_prompt()}
用户输入: {user_input}
请判断需要使用哪个技能（如果有），并以JSON格式输出：{{"skill": "技能名", "params": {{"content": "需要处理的文本内容"}} }}
如果不需要技能，输出 {{"skill": null}}。只输出JSON，不要有其他内容。"""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("skill"), result.get("params", {})
        except:
            return None, {}

    def _load_full_skill(self, skill_name):
        """按需加载完整 SKILL.md（渐进式加载第二步）"""
        if skill_name in self.loaded_skills:
            print(f"[Cache] 技能 '{skill_name}' 已加载，从缓存读取")
            return self.loaded_skills[skill_name]

        for meta in self.skills_meta:
            if meta["name"] == skill_name:
                skill_md = meta["path"] / "SKILL.md"
                full_content = skill_md.read_text(encoding='utf-8')
                self.loaded_skills[skill_name] = full_content
                print(f"[Loader] 首次加载技能 '{skill_name}'，内容长度 {len(full_content)} 字符")
                return full_content
        return None

    def _execute_script_sandbox(self, skill_name, params):
        """在沙箱中执行技能脚本（使用脚本绝对路径，工作目录为临时目录）"""
        for meta in self.skills_meta:
            if meta["name"] == skill_name:
                script_path = meta["path"] / "script.py"
                if not script_path.exists():
                    return f"错误：技能 {skill_name} 缺少 script.py"
                content = params.get("content", "")
                # 使用绝对路径运行脚本，工作目录为临时沙箱
                with tempfile.TemporaryDirectory() as tmpdir:
                    cmd = ["python", str(script_path.absolute()), content]
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=tmpdir,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            return result.stdout.strip()
                        else:
                            return f"脚本执行错误：{result.stderr}"
                    except subprocess.TimeoutExpired:
                        return "执行超时（10秒）"
        return f"未找到技能 {skill_name}"

    def run(self, user_input):
        print(f"\n用户: {user_input}")
        # 1. 用大模型判断技能和参数
        skill_name, params = self._select_skill_and_params(user_input)
        if not skill_name:
            # 没有匹配技能，直接让大模型回答
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": user_input}]
            )
            answer = response.choices[0].message.content
            print(f"助手: {answer}")
            return answer

        # 2. 按需加载完整技能指令（演示渐进式加载）
        full_skill = self._load_full_skill(skill_name)
        if not full_skill:
            print(f"助手: 无法加载技能 {skill_name}")
            return

        # 3. 沙箱执行脚本
        print(f"[Agent] 调用技能 '{skill_name}'，参数: {params}")
        output = self._execute_script_sandbox(skill_name, params)

        # 4. 返回执行结果
        final_response = f"技能 {skill_name} 执行完成:\n{output}"
        print(f"助手: {final_response}")
        return final_response


if __name__ == "__main__":
    agent = SkillAgent()
    # 首次运行：技能元数据已加载，但完整SKILL.md尚未加载
    print("=== 首次调用（将触发渐进加载）===")
    agent.run("请帮我统计以下文本的行数和单词数：Hello world\nThis is a test")

    print("\n=== 第二次调用同一技能（从缓存读取）===")
    agent.run("再统计这段：One line")