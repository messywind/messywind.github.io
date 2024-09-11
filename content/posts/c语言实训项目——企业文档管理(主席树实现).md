---
title: "c语言实训项目——企业文档管理(主席树实现)"
date: 2021-07-06 08:35:31
tags:
- 实训项目
categories:
- 杂项
code:
  maxShownLines: 11
---

```cpp
#include <bits/stdc++.h> //万能头 
#include <windows.h> //调用Sleep()需要 
using namespace std; //命名空间 
const int N = 1e5 + 5; //最大数组空间 
int version, idx, log_id, history_id; //version表示现在版本  idx表示id号 log_id表示日志id  history_id表示历史id
string Your_pos, Content, op; //op表示当前操作 Your_pos表示你的职位  Content表示内容
char line[N]; //文件每一行的数组 
FILE *in, *out; //文件指针 
struct Segment_Tree { //线段树结构体 
    int l, r; //l左儿子节点，r右儿子节点 
    string content; //每个节点存的内容 
} tr[N << 5]; //空间乘32防止溢出 
struct Chairman_Tree_Root { //主席树根结构体 
	int id, end; //id表示根的id， end表示每个版本有多少个文档 
} root[N];
void build(int &p, int l, int r) { //建树 
    p = ++ idx; //建立新节点 
    if (l == r) return ; //递归结束边界 
    int mid = l + r >> 1; //取中点 
    build(tr[p].l, l, mid), build(tr[p].r, mid + 1, r); //递归建树 
}
void insert(int &p, int pre, int pos, string x, int l, int r) { //主席树插入 
    p = ++ idx; //建立新版本 
    tr[p].l = tr[pre].l, tr[p].r = tr[pre].r; //复制旧版本的信息 
    tr[p].content = tr[pre].content; //复制旧版本的内容 
    if (l == r) { //新插内容 
        tr[p].content = x;
        return ;
    }
    int mid = l + r >> 1; //取中点 
    if (pos <= mid){ //递归插入
        insert(tr[p].l, tr[pre].l, pos, x, l, mid);
    } else {
        insert(tr[p].r, tr[pre].r, pos, x, mid + 1, r);
    }
}
string ask(int now, int pos, int l, int r) { //主席树查询，返回一个string类型字符串 
    if (l == r) return tr[now].content; //递归边界，返回位置为pos的查询结果 
    int mid = l + r >> 1;
    if (pos <= mid) { //递归查询 
        return ask(tr[now].l, pos, l, mid);
    } else {
        return ask(tr[now].r, pos, mid + 1, r);
    }
}
void Gui(); //预先定义主界面函数 
void Show_List(int Show_Version); //预先定义显示所有文档 
void Welcome() {
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t                        \t|\n");
	printf("\t\t|\t                        \t|\n");
	printf("\t\t|\t欢迎访问企业文档管理系统\t|\n");
	printf("\t\t|\t 作者:某某某1111某某某  \t|\n");
	printf("\t\t|\t                        \t|\n");
	printf("\t\t|\t                        \t|\n");
	printf("\t\t*---------------------------------------*\n");
} 
void back() { //返回 
	Sleep(1000); //延迟1秒 
	system("cls"); //清屏 
	Gui(); //显示主界面 
}
bool Empty_Check() { //文档判空 
	if (!root[version].end || !version) { //当前没有任何版本 或 此版本没有文档就说明为空 
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t文档为空!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
		back();
	}
	return 0;
}
void Show_History() { //显示历史版本 
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t当前的历史版本号有:\t|\n");
	printf("\t\t*---------------------------------------*\n\n");
	for (int i = 1; i <= version; i ++) {
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t版本%d:\t\t\t|\n", i);
		printf("\t\t*---------------------------------------*\n");
		Show_List(i); //调用展示第i个版本的所有文档就可以，节省代码量 
		cout << endl;
	}
	cout << endl;
}
bool Power_Check() { //权限判断 
	printf("输入你的职务(老板、员工):");
	getchar(), getline(cin, Your_pos);
	if (Your_pos == "老板") {
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t尊敬的老板，欢迎您!\t|\n");
		printf("\t\t*---------------------------------------*\n");
		return 1;
	} else if (Your_pos == "员工") {
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t没有权限这么做!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
		return 0;
	} else {
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t职务不合法!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
		return 0;
	}
}
void Show_List(int Show_Version) { //显示所有文档列表 
	Empty_Check(); //判空 
	for (int i = 1; i <= root[Show_Version].end; i ++) { //显示所有版本的所有文档 
		printf("\t\t*---------------------------------------*\n\t\t");
		cout << i << "." << ask(root[Show_Version].id, i, 1, N - 1) << endl; //查询第i个版本的文档 
		Sleep(100);
	}
	printf("\t\t*---------------------------------------*\n");
	cout << endl;
}
void History_Roll_Back() { //历史记录查看和回滚 
	Empty_Check(); //判空 
	Show_History(); //显示历史记录 
	printf("\n请输入你要回滚的历史版本号:");
	cin >> history_id;
	if (history_id > version || history_id <= 0) { //判断版本号合法 
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t版本号不合法!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
	} else {
		version = history_id;
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t回滚成功!当前的版本为:%d\t\t|\n", history_id);
		printf("\t\t*---------------------------------------*\n");
	}
	back();
}
void Update_Log() { //上传日志 
	printf("请输入你要上传的日志内容:");
	getchar(), getline(cin, Content);
	version ++, root[version].end = root[version - 1].end + 1; //新建版本 
	insert(root[version].id, root[version - 1].id, root[version].end, Content, 1, N - 1); //更新此版本 
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t上传成功!\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	back();
}
void Modify_Log() { //修改日志 
	Empty_Check(); //判空 
	Show_List(version); //显示文档列表 
	printf("请输入你要修改的日志号:");
	cin >> log_id;
	if (log_id > root[version].end || log_id <= 0) { //判断日志号合法 
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t该日志不存在!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
	} else {
		printf("编号为%d的内容为:", log_id);
		cout << ask(root[version].id, log_id, 1, N - 1) << endl; //查询id 
		printf("\n请输入你要修改的日志内容:");
		cin >> Content;
		version ++, root[version].end = root[version - 1].end; //新建版本 
		insert(root[version].id, root[version - 1].id, log_id, Content, 1, N - 1); //修改 
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t修改成功!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
	}
	back();
}
void Delete_Log() { //删除日志 
	if (Power_Check()) { //权限判断 
		Show_List(version); //显示文档列表 
		printf("请输入你要删除的日志号:");
		cin >> log_id;
		if (log_id > root[version].end || log_id <= 0) { //判断日志号合法 
			printf("\t\t*---------------------------------------*\n");
			printf("\t\t|\t\t该日志不存在!\t\t|\n");
			printf("\t\t*---------------------------------------*\n");
		} else {
			printf("\t\t*---------------------------------------*\n");
			printf("\t\t|\t\t删除成功!\t\t|\n");
			printf("\t\t*---------------------------------------*\n");
			version ++, root[version].end = root[version - 1].end; //新建版本 
			insert(root[version].id,root[version - 1].id, log_id, "该文档已被删除!", 1, N - 1); //删除文档，用“该文档已被删除”来替换 
		}
	}
	back(); 
}
void Gui() {
	Welcome();
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t          菜单            \t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t1.展示文档列表\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t2.历史版本回滚\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t3.上传日志\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t4.修改日志\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t5.删除日志\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t|\t\t6.退出\t\t\t|\n");
	printf("\t\t*---------------------------------------*\n");
	printf("\t\t\t\t请输入操作:");
	cin >> op;
	system("cls");
	Welcome();
	if (op == "1") {
		Show_List(version);  //显示当前版本，也就是版本为version的版本 
		back();
	} else if (op == "2") {
		History_Roll_Back();
	} else if (op == "3") {
		Update_Log();
	} else if (op == "4") {
		Modify_Log();
	} else if (op == "5") {
		Delete_Log();
	} else if (op == "6") {
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t谢谢!\t\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
		for (int i = 1; i <= root[version].end; i ++) { //把现有文件导入到out.txt 
			string OutLine = ask(root[version].id, i, 1, N - 1) + '\n'; //拿出来每个节点 
			const char *output = OutLine.c_str(); //string转char 
			fputs(output, out); //输出到文件 
		}
		Sleep(500);
		exit(0);
	} else { //判操作不合法 
		printf("\t\t*---------------------------------------*\n");
		printf("\t\t|\t\t操作不合法!\t\t|\n");
		printf("\t\t*---------------------------------------*\n");
		back();
	}
}
signed main() { //主函数 
	system("color 2e"); //系统颜色 
	build(root[0].id, 1, N - 1); //线段树建树 
	in = fopen("in.txt", "r"), out = fopen("out.txt", "w"); //打开文件in.txt, out.txt
	while (fgets(line, N, in) != NULL) { //读入每一行 
		int len = strlen(line);
		line[len - 1] = '\0'; //删去末尾回车 
		version ++, root[version].end = root[version - 1].end + 1; //预处理版本
		insert(root[version].id, root[version - 1].id, root[version].end, line, 1, N - 1); //插入一行到版本中 
	}
	Gui(); //显示欢迎界面 
}
```

