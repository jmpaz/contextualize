import requests
import re
from datetime import datetime


class LinearClient:
    def __init__(self, access_token):
        self.access_token = access_token
        self.base_url = "https://api.linear.app/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def execute_query(self, query, variables=None):
        data = {"query": query, "variables": variables or {}}
        response = requests.post(self.base_url, headers=self.headers, json=data)
        return response.json()

    def get_issue(self, issue_id):
        query = """
            query ($issueId: String!) {
              issue(id: $issueId) {
                id identifier title description
                state { id name }
                priority estimate
                assignee { id name }
                labels { nodes { id name } }
                project { id name }
                relations {
                  nodes {
                    id type
                    relatedIssue { id identifier title }
                  }
                }
                comments {
                  nodes {
                    id body
                    parent { id }
                    user { id name }
                    createdAt
                  }
                }
              }
            }
        """
        variables = {"issueId": issue_id}
        response_data = self.execute_query(query, variables)
        issue_data = response_data["data"]["issue"]
        return Issue(issue_data)


class Issue:
    def __init__(self, data):
        self.id = data["id"]
        self.identifier = data["identifier"]
        self.title = data["title"]
        self.description = data["description"]
        self.priority = data["priority"]
        self.estimate = data["estimate"]
        self.assignee = data["assignee"]
        self.labels = [label["name"] for label in data["labels"]["nodes"]]
        self.project = data["project"]
        self.state = data["state"]
        self.relations = data["relations"]
        self.comments = self._process_comments(data["comments"]["nodes"])

    def _process_comments(self, comments_data):
        comments = []
        for comment_data in comments_data:
            comment = {
                "id": comment_data["id"],
                "body": comment_data["body"],
                "user": comment_data["user"]["name"],
                "created_at": datetime.strptime(
                    comment_data["createdAt"], "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "parent_id": comment_data["parent"]["id"]
                if comment_data["parent"]
                else None,
            }
            comments.append(comment)
        return sorted(comments, key=lambda c: c["created_at"])

    def _format_markdown_body(self, text):
        text = re.sub(r"~(.*?)~", r"~~\1~~", text)
        text = re.sub(r"\n\* (.*)", r"\n- \1", text)
        text = re.sub(r"\*(.*?)\*", r"_\1_", text)
        text = re.sub(r"\n\n", r"\n", text)
        text = re.sub(r"\!\[(.*?)\]\((.*?)\)", r"\\[\1]", text)
        return text

    def to_markdown(self, include_properties=None):
        if include_properties is None:
            include_properties = {}

        md = f"# {self.identifier}: {self.title}\n\n"

        if self.description:
            md += f"{self._format_markdown_body(self.description)}\n\n\n"

        properties = self._get_properties(include_properties)
        if properties:
            md += "## Properties\n\n```yaml\n"
            md += "\n".join(f"{prop}: {value}" for prop, value in properties.items())
            md += "\n```\n\n\n"

        if self.comments:
            md += "## Notes\n\n"
            for comment in self.comments:
                if not comment["parent_id"]:
                    md += self._format_comment(comment)
                    replies = [
                        c for c in self.comments if c["parent_id"] == comment["id"]
                    ]
                    for reply in replies:
                        md += self._format_comment(reply, is_reply=True)
                    md += "\n"

        return md

    def _format_comment(self, comment, is_reply=False):
        comment_md = f"{'####' if is_reply else '###'} {comment['id'][:5]}\n"
        comment_md += f"{self._format_markdown_body(comment['body'])}\n\n"
        return comment_md

    def _get_properties(self, include_properties):
        if include_properties is None:
            include_properties = []

        priority_mapping = {1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}
        estimate_mapping = {1: "XS", 2: "S", 3: "M", 4: "L", 5: "XL"}

        property_configs = {
            "state": ("State", self.state.get("name")),
            "priority": (
                "Priority",
                priority_mapping.get(self.priority, None),
            ),
            "estimate": (
                "Estimate",
                estimate_mapping.get(self.estimate, None),
            ),
            "assignee": (
                "Assignee",
                self.assignee.get("name") if self.assignee else None,
            ),
            "labels": ("Labels", ", ".join(self.labels) if self.labels else None),
            "project": ("Project", self.project.get("name")),
            "relations": ("Relations", self._format_relations()),
        }

        properties = {
            config[0].lower(): config[1]
            for prop, config in property_configs.items()
            if prop in include_properties and config[1]
        }

        return properties

    def _format_relations(self):
        if (
            not self.relations
            or "nodes" not in self.relations
            or not self.relations["nodes"]
        ):
            return None
        relations_str = "\n" + "\n".join(
            f"  - {node['relatedIssue']['title']} ({node['relatedIssue']['identifier']})"
            for node in self.relations["nodes"]
        )
        return relations_str
