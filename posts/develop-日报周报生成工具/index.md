# 日报/周报生成工具


{{&lt; WeeklyReport &gt;}}

## 代码：(Vue3 &#43; TypeScript)

```html
&lt;template&gt;
  &lt;div class=&#34;app-container&#34;&gt;
    &lt;h2 class=&#34;page-title&#34;&gt;日报/周报生成工具&lt;/h2&gt;
    &lt;div class=&#34;card-container&#34;&gt;
      &lt;!-- 日期卡片按周五、周六、周一、周二、周三、周四顺序排列 --&gt;
      &lt;div v-for=&#34;day in days&#34; :key=&#34;day.id&#34; class=&#34;day-card&#34;&gt;
        &lt;div class=&#34;card-header&#34;&gt;
          &lt;h2 class=&#34;day-name&#34;&gt;{{ day.name }}&lt;/h2&gt;
          &lt;button class=&#34;add-task-btn&#34; @click=&#34;openAddTaskModal(day.id)&#34;&gt;&#43; 添加任务&lt;/button&gt;
        &lt;/div&gt;
        &lt;div class=&#34;task-list&#34;&gt;
          &lt;div v-for=&#34;(task, index) in day.tasks&#34; :key=&#34;index&#34; class=&#34;task-item&#34;&gt;
            &lt;div class=&#34;task-content&#34;&gt;
              &lt;div class=&#34;task-main&#34;&gt;
                &lt;span class=&#34;task-description&#34;&gt;{{ task.description }}&lt;/span&gt;
              &lt;/div&gt;
              &lt;div class=&#34;task-meta&#34;&gt;
                &lt;span class=&#34;task-project&#34;&gt;{{ task.project }}&lt;/span&gt;
                &lt;span class=&#34;task-hours&#34;&gt;工作时长: {{ task.workHours }}h&lt;/span&gt;
                &lt;span class=&#34;task-overtime&#34;&gt;加班: {{ task.overtimeHours }}h&lt;/span&gt;
              &lt;/div&gt;
            &lt;/div&gt;
            &lt;div class=&#34;task-actions&#34;&gt;
              &lt;button @click=&#34;openEditTaskModal(day.id, index)&#34;&gt;编辑&lt;/button&gt;
              &lt;button @click=&#34;deleteTask(day.id, index)&#34;&gt;删除&lt;/button&gt;
            &lt;/div&gt;
          &lt;/div&gt;
          &lt;div v-if=&#34;day.tasks.length === 0&#34; class=&#34;empty-task&#34;&gt;
            暂无任务，点击&#34;添加任务&#34;开始
          &lt;/div&gt;
        &lt;/div&gt;
      &lt;/div&gt;
    &lt;/div&gt;
    &lt;button @click=&#34;generateDailyReport&#34; class=&#34;generate-report-btn daily-report-btn&#34;&gt;生成日报&lt;/button&gt;
    &lt;button @click=&#34;generateReport&#34; class=&#34;generate-report-btn&#34;&gt;生成周报&lt;/button&gt;
    &lt;button @click=&#34;clearAllTasks&#34; class=&#34;generate-report-btn clear-tasks-btn&#34;&gt;清空所有任务&lt;/button&gt;

    &lt;div v-if=&#34;dailyReportResult&#34; class=&#34;daily-report-result&#34;&gt;
      &lt;div v-for=&#34;(report, reportIndex) in dailyReportResult&#34; :key=&#34;reportIndex&#34; class=&#34;daily-report-day&#34;&gt;
        &lt;h2&gt;{{ report.dayName }}日报&lt;/h2&gt;
        &lt;div class=&#34;daily-report-section&#34;&gt;
          &lt;h3&gt;一、今日计划：&lt;/h3&gt;
          &lt;p v-for=&#34;plan in report.plans&#34; :key=&#34;plan.id&#34;&gt;
            {{ plan.id }}. {{ plan.description }}；工作用时{{ plan.workHours }}小时
          &lt;/p&gt;
        &lt;/div&gt;
        &lt;div class=&#34;daily-report-section&#34;&gt;
          &lt;h3&gt;二、完成情况：&lt;/h3&gt;
          &lt;p v-for=&#34;completion in report.completions&#34; :key=&#34;completion.id&#34;&gt;
            {{ completion.id }}. {{ completion.status }}，用时 {{ completion.hours }} 小时。
          &lt;/p&gt;
        &lt;/div&gt;
        &lt;div v-if=&#34;report.overtime.totalHours &gt; 0&#34; class=&#34;daily-report-section&#34;&gt;
          &lt;h3&gt;三、加班情况：加班 {{ report.overtime.totalHours }} 小时。&lt;/h3&gt;
          &lt;p v-for=&#34;task in report.overtime.tasks&#34; :key=&#34;task.id&#34;&gt;
            {{ task.id }}. {{ task.description }}；加班 {{ task.hours }} 小时。
          &lt;/p&gt;
        &lt;/div&gt;
      &lt;/div&gt;
    &lt;/div&gt;

    &lt;div v-if=&#34;reportResult&#34; class=&#34;report-result&#34;&gt;
      &lt;h2&gt;周报&lt;/h2&gt;
      &lt;div class=&#34;report-projects&#34;&gt;
        &lt;div v-for=&#34;(project, index) in reportResult.projects&#34; :key=&#34;project.name&#34; class=&#34;project-section&#34;&gt;
          &lt;h3&gt;{{ numberToChinese(index &#43; 1) }}、{{ project.name }}：&lt;/h3&gt;
          &lt;p v-for=&#34;(task, taskIndex) in project.tasks&#34; :key=&#34;taskIndex&#34;&gt;
            {{ `${taskIndex &#43; 1}. ${task.description}${(task.workHours &gt; 0 || task.overtimeHours &gt; 0) ? &#39;，&#39; :
              &#39;&#39;}${[task.workHours &gt; 0 ? `工作时长: ${task.workHours} 小时` : &#39;&#39;, task.overtimeHours &gt; 0 ? `加班时长:
            ${task.overtimeHours} 小时` : &#39;&#39;].filter(Boolean).join(&#39;，&#39;)}。` }}
          &lt;/p&gt;
        &lt;/div&gt;
      &lt;/div&gt;
      &lt;div class=&#34;report-summary&#34;&gt;
        本周总工作时长{{ reportResult.totalWorkHours }} &#43; {{ reportResult.totalOvertimeHours }}小时。
      &lt;/div&gt;
    &lt;/div&gt;

    &lt;!-- 添加/编辑任务模态框 --&gt;
    &lt;div v-if=&#34;isModalOpen&#34; class=&#34;modal-overlay&#34;&gt;
      &lt;div class=&#34;modal-content&#34;&gt;
        &lt;h3 class=&#34;modal-title&#34;&gt;{{ currentTaskId === null ? &#39;添加任务&#39; : &#39;编辑任务&#39; }}&lt;/h3&gt;
        &lt;form&gt;
          &lt;div class=&#34;form-group&#34;&gt;
            &lt;label&gt;任务描述:&lt;/label&gt;
            &lt;input type=&#34;text&#34; v-model=&#34;newTask.description&#34; required&gt;
          &lt;/div&gt;
          &lt;div class=&#34;form-group&#34;&gt;
            &lt;label&gt;项目名称:&lt;/label&gt;
            &lt;input type=&#34;text&#34; v-model=&#34;newTask.project&#34; required&gt;
          &lt;/div&gt;
          &lt;div class=&#34;form-row&#34;&gt;
            &lt;div class=&#34;form-group half&#34;&gt;
              &lt;label&gt;工作时长 (小时):&lt;/label&gt;
              &lt;input type=&#34;number&#34; v-model.number=&#34;newTask.workHours&#34; min=&#34;0&#34; step=&#34;0.5&#34; required&gt;
            &lt;/div&gt;
            &lt;div class=&#34;form-group half&#34;&gt;
              &lt;label&gt;加班时长 (小时):&lt;/label&gt;
              &lt;input type=&#34;number&#34; v-model.number=&#34;newTask.overtimeHours&#34; min=&#34;0&#34; step=&#34;0.5&#34; required&gt;
            &lt;/div&gt;
          &lt;/div&gt;
          &lt;div class=&#34;modal-buttons&#34;&gt;
            &lt;button type=&#34;button&#34; @click=&#34;closeModal&#34;&gt;取消&lt;/button&gt;
            &lt;button type=&#34;submit&#34; @click=&#34;saveTask&#34;&gt;保存&lt;/button&gt;
          &lt;/div&gt;
        &lt;/form&gt;
      &lt;/div&gt;
    &lt;/div&gt;
  &lt;/div&gt;
&lt;/template&gt;

&lt;script setup lang=&#34;ts&#34;&gt;
import { ref, reactive } from &#39;vue&#39;;

// 定义任务类型
interface Task {
  description: string;
  project: string;
  workHours: number;
  overtimeHours: number;
}

// 定义日期类型
interface Day {
  id: number;
  name: string;
  tasks: Task[];
}

// 从localStorage加载任务数据
const loadFromLocalStorage = (): Day[] =&gt; {
  const savedDays = localStorage.getItem(&#39;weeklyTasks&#39;);
  if (savedDays) {
    try {
      return JSON.parse(savedDays);
    } catch (e) {
      console.error(&#39;Failed to parse saved tasks&#39;, e);
    }
  }
  // 默认初始化数据
  return [
    { id: 1, name: &#39;周五&#39;, tasks: [] },
    { id: 2, name: &#39;周六&#39;, tasks: [] },
    { id: 3, name: &#39;周一&#39;, tasks: [] },
    { id: 4, name: &#39;周二&#39;, tasks: [] },
    { id: 5, name: &#39;周三&#39;, tasks: [] },
    { id: 6, name: &#39;周四&#39;, tasks: [] },
  ];
};

// 保存任务数据到localStorage
const saveToLocalStorage = () =&gt; {
  localStorage.setItem(&#39;weeklyTasks&#39;, JSON.stringify(days));
};

// 初始化日期数据
const days = reactive&lt;Day[]&gt;(loadFromLocalStorage());

// 模态框状态
const isModalOpen = ref(false);
const currentDayId = ref&lt;number | null&gt;(null);
const currentTaskId = ref&lt;number | null&gt;(null);

const reportResult = ref&lt;null | {
  totalWorkHours: number;
  totalOvertimeHours: number;
  projects: Array&lt;{
    name: string;
    tasks: Array&lt;{
      index: number;
      description: string;
      workHours: number;
      overtimeHours: number;
    }&gt;;
  }&gt;
}&gt;(null);
const dailyReportResult = ref&lt;DailyReport[] | null&gt;(null);

// 新任务/编辑任务数据
const newTask = reactive&lt;Task&gt;({
  description: &#39;&#39;,
  project: &#39;&#39;,
  workHours: 0,
  overtimeHours: 0,
});

// 打开添加任务模态框
function openAddTaskModal(dayId: number) {
  currentDayId.value = dayId;
  currentTaskId.value = null;
  // 重置表单
  newTask.description = &#39;&#39;;
  newTask.project = &#39;&#39;;
  newTask.workHours = 0;
  newTask.overtimeHours = 0;
  isModalOpen.value = true;
}

// 打开编辑任务模态框
function openEditTaskModal(dayId: number, taskIndex: number) {
  currentDayId.value = dayId;
  currentTaskId.value = taskIndex;
  const day = days.find(d =&gt; d.id === dayId);
  if (day &amp;&amp; day.tasks[taskIndex]) {
    const task = day.tasks[taskIndex];
    // 填充表单
    newTask.description = task.description;
    newTask.project = task.project;
    newTask.workHours = task.workHours;
    newTask.overtimeHours = task.overtimeHours;
    isModalOpen.value = true;
  }
}

// 关闭模态框
function closeModal() {
  isModalOpen.value = false;
}

// 保存任务
function saveTask() {
  if (!currentDayId.value) return;

  const day = days.find(d =&gt; d.id === currentDayId.value);
  if (!day) return;

  if (currentTaskId.value === null) {
    // 添加新任务
    day.tasks.push({
      description: newTask.description,
      project: newTask.project,
      workHours: newTask.workHours,
      overtimeHours: newTask.overtimeHours,
    });
  } else {
    // 编辑现有任务
    day.tasks[currentTaskId.value] = {
      description: newTask.description,
      project: newTask.project,
      workHours: newTask.workHours,
      overtimeHours: newTask.overtimeHours,
    };
  }

  closeModal();
  saveToLocalStorage();
}

// 清空所有任务
import { ElMessageBox, ElMessage } from &#39;element-plus&#39;;

async function clearAllTasks() {
  try {
    await ElMessageBox.confirm(
      &#39;确定要清空所有任务吗？此操作不可恢复。&#39;,
      &#39;警告&#39;,
      {
        confirmButtonText: &#39;确定&#39;,
        cancelButtonText: &#39;取消&#39;,
        type: &#39;warning&#39;,
      }
    );
    days.forEach(day =&gt; {
      day.tasks = [];
    });
    saveToLocalStorage();
    ElMessage.success(&#39;所有任务已成功清空&#39;);
  } catch {
    ElMessage.info(&#39;已取消清空操作&#39;);
  }
}

// 删除任务
function deleteTask(dayId: number, taskIndex: number) {
  const day = days.find(d =&gt; d.id === dayId);
  if (day) {
    day.tasks.splice(taskIndex, 1);
    saveToLocalStorage();
  }
}

// 定义日报数据接口
interface DailyReport {
  dayName: string;
  plans: Array&lt;{
    id: number;
    description: string;
    workHours: number;
  }&gt;;
  completions: Array&lt;{
    id: number;
    status: string;
    hours: number;
  }&gt;;
  overtime: {
    totalHours: number;
    tasks: Array&lt;{
      id: number;
      description: string;
      hours: number;
    }&gt;;
  };
}

// 生成日报
function generateDailyReport() {
  // 定义日期顺序：周五、周六、周一、周二、周三、周四
  const dayOrder = [&#39;周五&#39;, &#39;周六&#39;, &#39;周一&#39;, &#39;周二&#39;, &#39;周三&#39;, &#39;周四&#39;];
  const dailyReports: DailyReport[] = [];

  dayOrder.forEach(dayName =&gt; {
    const day = days.find(d =&gt; d.name === dayName);
    if (!day || day.tasks.length === 0) return;

    // 构建日报数据
    const report = {
      dayName,
      plans: day.tasks.map((task, index) =&gt; ({
        id: index &#43; 1,
        description: task.description,
        workHours: task.workHours
      })),
      completions: day.tasks.map((task, index) =&gt; ({
        id: index &#43; 1,
        status: &#39;完成&#39;,
        hours: task.workHours &#43; task.overtimeHours
      })),
      overtime: {
        totalHours: day.tasks.reduce((sum, task) =&gt; sum &#43; task.overtimeHours, 0),
        tasks: day.tasks.filter(task =&gt; task.overtimeHours &gt; 0).map((task, index) =&gt; ({
          id: index &#43; 1,
          description: task.description,
          hours: task.overtimeHours
        }))
      }
    };

    dailyReports.push(report);
  });

  dailyReportResult.value = dailyReports;
}

// 生成周报
function numberToChinese(num: number) {
  const chineseNumbers = [&#39;&#39;, &#39;一&#39;, &#39;二&#39;, &#39;三&#39;, &#39;四&#39;, &#39;五&#39;, &#39;六&#39;, &#39;七&#39;, &#39;八&#39;, &#39;九&#39;, &#39;十&#39;];
  return chineseNumbers[num] || num;
}

function generateReport() {
  let totalWorkHours = 0;
  let totalOvertimeHours = 0;
  const projects: Record&lt;string, Array&lt;{
    description: string;
    workHours: number;
    overtimeHours: number;
    index?: number;
  }&gt;&gt; = {};

  // 收集所有任务并按项目分类
  days.forEach(day =&gt; {
    day.tasks.forEach(task =&gt; {
      const isSaturday = day.name === &#39;周六&#39;;
      const workHours = isSaturday ? 0 : task.workHours;
      const overtimeHours = isSaturday ? (task.workHours &#43; task.overtimeHours) : task.overtimeHours;

      totalWorkHours &#43;= workHours;
      totalOvertimeHours &#43;= overtimeHours;

      if (!projects[task.project]) {
        projects[task.project] = [];
      }
      projects[task.project].push({
        description: task.description,
        workHours: workHours,
        overtimeHours: overtimeHours
      });
    });
  });

  // 为所有任务统一编号
  let taskIndex = 0;
  const projectList = Object.entries(projects).map(([name, tasks]) =&gt; {
    tasks.forEach(task =&gt; {
      taskIndex&#43;&#43;;
      task.index = taskIndex;
    });
    return { name, tasks };
  });

  // 更新报告结果
  // 修复类型不匹配问题
  reportResult.value = {
    totalWorkHours,
    totalOvertimeHours,
    projects: projectList.map(project =&gt; ({
      name: project.name,
      tasks: project.tasks.map(task =&gt; ({
        index: task.index || 0,
        description: task.description,
        workHours: task.workHours,
        overtimeHours: task.overtimeHours
      }))
    }))
  } as typeof reportResult.value;
}
&lt;/script&gt;

&lt;style&gt;
:root {
  --primary-color: #42B983;
  --primary-light: #e6f7ef;
  --primary-dark: #359469;
  --text-on-primary: #333333;
  --secondary-color: #3498db;
}
&lt;/style&gt;

&lt;style scoped&gt;
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: &#39;Arial&#39;, sans-serif;
}

.page-title {
  text-align: center;
  color: #42B983;
  margin-bottom: 30px;
}

.card-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}

.day-card {
  background-color: white;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: transform 0.2s;
}

.day-card:hover {
  transform: translateY(-5px);
}

.card-header {
  background-color: var(--primary-color);
  color: var(--text-on-primary);
  padding: 15px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.day-name {
  margin: 0;
  font-size: 1.2rem;
}

.add-task-btn {
  background-color: white;
  color: #42b983;
  border: none;
  border-radius: 20px;
  padding: 5px 15px;
  cursor: pointer;
  font-weight: bold;
}

.task-list {
  padding: 15px;
  max-height: 400px;
  overflow-y: auto;
}

.task-item {
  background-color: #f9f9f9;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-content {
  flex: 1;
}

.task-main {
  margin-bottom: 5px;
}

.task-description {
  font-size: 1rem;
  color: #333;
}

.task-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 0.8rem;
  color: #666;
}

.task-project {
  background-color: var(--primary-light);
  padding: 2px 8px;
  border-radius: 12px;
  color: var(--primary-dark);
}

.task-actions {
  display: flex;
  gap: 8px;
}

.task-actions button {
  background: none;
  border: none;
  cursor: pointer;
  color: var(--primary-color);
}

.task-actions button:last-child {
  color: #e74c3c;
}

.empty-task {
  text-align: center;
  padding: 20px;
  color: #999;
  font-style: italic;
}

.generate-btn {
  background-color: var(--primary-color);
  color: var(--text-on-primary);
  border: none;
  border-radius: 5px;
  padding: 12px 30px;
  font-size: 1rem;
  cursor: pointer;
  display: inline-block;
  margin-right: 10px;
  margin-top: 20px;
}

.daily-report-btn {
  margin-left: 10px;
  background-color: var(--secondary-color);
}

.clear-tasks-btn {
  background-color: #ff4444;
  margin-left: 10px;
}

.clear-tasks-btn:hover {
  background-color: #cc0000;
}

.generate-report-btn {
  background-color: var(--primary-color);
  color: var(--text-on-primary);
  border: none;
  border-radius: 5px;
  padding: 12px 30px;
  font-size: 1rem;
  cursor: pointer;
  display: inline-block;
  margin-right: 10px;
  margin-top: 20px;
}

.daily-report-result {
  margin-top: 20px;
  padding: 15px;
  background-color: var(--primary-light);
  border-radius: 8px;
  color: var(--text-on-primary);
}

.daily-report-day {
  margin-bottom: 25px;
  padding: 15px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
  color: #333333;
  line-height: 1.6;
}

.daily-report-day h2 {
  color: var(--primary-color);
  margin-top: 0;
  margin-bottom: 15px;
}

.daily-report-day:last-child {
  border-bottom: none;
}

.daily-report-section {
  margin-bottom: 15px;
}

.daily-report-section h3 {
  color: var(--primary-dark);
  border-bottom: 1px solid var(--primary-light);
  padding-bottom: 5px;
  margin-top: 15px;
  margin-bottom: 10px;
}

.daily-report-section ul {
  margin-left: 20px;
}

.daily-report-section li {
  margin-bottom: 5px;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background-color: white;
  border-radius: 10px;
  width: 90%;
  max-width: 500px;
  padding: 20px;
}

.modal-title {
  margin-top: 0;
  color: #333;
}

.form-group {
  margin-bottom: 15px;
}

.form-row {
  display: flex;
  gap: 10px;
}

.half {
  flex: 1;
}

label {
  display: block;
  margin-bottom: 5px;
  color: #666;
}

input {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.modal-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 20px;
}

.modal-buttons button {
  padding: 8px 15px;
  border-radius: 4px;
  cursor: pointer;
}

.modal-buttons button:first-child {
  background-color: #f2f2f2;
  border: 1px solid #ddd;
}

.modal-buttons button:last-child {
  background-color: var(--primary-color);
  color: var(--text-on-primary);
  border: none;
}

.report-result {
  margin-top: 20px;
  padding: 20px;
  background-color: var(--primary-color);
  border-radius: 8px;
  color: var(--text-on-primary);
}

.report-summary {
  font-weight: bold;
  margin-bottom: 15px;
  color: white;
}

.report-projects {
  margin-top: 15px;
}

.project-section {
  margin-bottom: 15px;
}

.project-section h3 {
  margin-bottom: 5px;
  color: var(--primary-light);
}

.project-section ul {
  padding-left: 20px;
}

.project-section li {
  margin-bottom: 5px;
}
&lt;/style&gt;
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/develop-%E6%97%A5%E6%8A%A5%E5%91%A8%E6%8A%A5%E7%94%9F%E6%88%90%E5%B7%A5%E5%85%B7/  

