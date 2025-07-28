# Vue3 项目打包为 Shortcodes 流程


shortcodes 仅支持单体 HTML 文件，所以需要一个 Vue 组件让他打包为单体文件。

在 Vue 项目中安装 `vite-plugin-singlefile` 组件：

```shell
npm install vite-plugin-singlefile --save-dev
```

在 `vite.config.ts` 配置：

```javascript
import { defineConfig } from &#39;vite&#39;;
import vue from &#39;@vitejs/plugin-vue&#39;;
import { viteSingleFile } from &#39;vite-plugin-singlefile&#39;;

export default defineConfig({
plugins: [vue(), viteSingleFile()],
});
```

打包：

```shell
npm run build
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/blog-vue3-%E6%89%93%E5%8C%85-shortcodes/  

