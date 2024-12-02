# 创想兵团一枪伤害计算


{{&lt; calc2 &gt;}}

## 代码
```vue
&lt;template&gt;
  &lt;el-form
      ref=&#34;formRef&#34;
      style=&#34;max-width: 600px&#34;
      :model=&#34;info&#34;
      :rules=&#34;rules&#34;
      label-width=&#34;auto&#34;
      status-icon
  &gt;
    &lt;el-form-item label=&#34;武器&#34; prop=&#34;weapon&#34;&gt;
      &lt;el-select v-model=&#34;info.weapon&#34; value-key=&#34;damage&#34; placeholder=&#34;选择武器&#34; style=&#34;width: 300px&#34;&gt;
        &lt;el-option
            v-for=&#34;item in weapons&#34;
            :key=&#34;item.damage&#34;
            :label=&#34;item.name &#43; &#39; 伤害:&#39; &#43; item.damage&#34;
            :value=&#34;item&#34;
        &gt;
          &lt;span style=&#34;float: left&#34;&gt;{{ item.name }}&lt;/span&gt;
          &lt;span style=&#34;
            float: right;
            color: var(--el-text-color-secondary);
            font-size: 13px;
          &#34;&gt;
            伤害:{{ item.damage }}
          &lt;/span&gt;
        &lt;/el-option&gt;
      &lt;/el-select&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;武器强化等级&#34; prop=&#34;level&#34;&gt;
      &lt;el-input-number v-model=&#34;info.level&#34; :min=&#34;0&#34; :max=&#34;20&#34; @change=&#34;handleChange&#34; /&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;己方破甲&#34; prop=&#34;breakArmor&#34;&gt;
      &lt;el-input-number v-model=&#34;info.breakArmor&#34; :min=&#34;0&#34; :max=&#34;40&#34; @change=&#34;handleChange&#34; /&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;对方护甲&#34; prop=&#34;Armor&#34;&gt;
      &lt;el-input-number v-model=&#34;info.Armor&#34; :min=&#34;0&#34; :max=&#34;400&#34; @change=&#34;handleChange&#34; /&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;伤害类型&#34; prop=&#34;Critical&#34;&gt;
      &lt;el-radio-group v-model=&#34;info.Critical&#34;&gt;
        &lt;el-radio :value=&#34;1&#34;&gt;普通&lt;/el-radio&gt;
        &lt;el-radio :value=&#34;2.5&#34;&gt;暴击(致命打击)&lt;/el-radio&gt;
        &lt;el-radio :value=&#34;3&#34;&gt;狼暴(狼卡 &#43; 致命打击)&lt;/el-radio&gt;
      &lt;/el-radio-group&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;改造伤害&#34; prop=&#34;addDamage&#34;&gt;
      &lt;el-input-number v-model=&#34;info.addDamage&#34; :min=&#34;0&#34; :max=&#34;43&#34; @change=&#34;handleChange&#34; /&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item label=&#34;伤害浮动百分比&#34; prop=&#34;addDamage&#34;&gt;
      &lt;el-input-number v-model=&#34;info.damageFloat&#34; :min=&#34;-6&#34; :max=&#34;6&#34; @change=&#34;handleChange&#34; /&gt;
    &lt;/el-form-item&gt;

    &lt;el-form-item&gt;
      &lt;el-button type=&#34;primary&#34; @click=&#34;calc(formRef)&#34;&gt;计算&lt;/el-button&gt;
    &lt;/el-form-item&gt;
  &lt;/el-form&gt;
  &lt;el-text class=&#34;mx-1&#34; type=&#34;danger&#34; v-if=&#34;show&#34;&gt;伤害数值：{{ damage }}&lt;/el-text&gt;
&lt;/template&gt;

&lt;script lang=&#34;ts&#34; setup&gt;
import { ref, reactive } from &#39;vue&#39;
import type { FormInstance } from &#39;element-plus&#39;

const formRef = ref&lt;FormInstance&gt;()

const info = reactive({
  weapon: undefined,
  level: 0,
  breakArmor: 0,
  Armor: 0,
  Critical: undefined,
  addDamage: 0,
  damageFloat: 0
})

const rules = reactive({
  weapon: [{
      required: true,
      message: &#39;请选择武器&#39;,
      trigger: &#39;change&#39;,
  }],
  Critical: [{
      required: true,
      message: &#39;请选择伤害类型&#39;,
      trigger: &#39;change&#39;,
  }]
})

const weapons = [{
    name: &#39;破心Ⅱ&#39;,
    damage: 1070,
  }, {
    name: &#39;祥云自由之鹰Ⅳ&#39;,
    damage: 1012,
  }, {
    name: &#39;DSR-7&#39;,
    damage: 1010,
  }, {
    name: &#39;祥云自由之鹰Ⅲ&#39;,
    damage: 1009,
  }, {
    name: &#39;音障突破/魔龙之吻Ⅲ&#39;,
    damage: 1008,
  }, {
    name: &#39;破心-夏季PK特别版&#39;,
    damage: 1003,
  }, {
    name: &#39;破心&#39;,
    damage: 995,
  }, {
    name: &#39;S224EV-DZ&#39;,
    damage: 987,
  }
]

function Ar (x) {
  return 29.11 * Math.log(x &#43; 170.83) - 149.86;
}

const show = ref(false)
const damage = ref(0)

const calc = async (formRef) =&gt; {
  await formRef.validate((valid) =&gt; {
    if (valid) {
      let addDamage = info.level * info.weapon.damage * 0.01;
      addDamage = parseFloat(addDamage.toFixed(2));

      let weaponDamage = info.Critical * (info.weapon.damage &#43; addDamage &#43; info.addDamage);
      weaponDamage *= parseFloat(((100 &#43; info.damageFloat) * 0.01).toFixed(2))
      weaponDamage = parseFloat(weaponDamage.toFixed(2));

      let armour = (100.0 - Ar(Math.max(info.Armor - info.breakArmor, 0.0))) * 0.01;
      armour = parseFloat(armour.toFixed(2));

      damage.value = (armour * weaponDamage).toFixed(2);
      show.value = true;
    }
  })
}
&lt;/script&gt;

```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/game%E5%88%9B%E6%83%B3%E5%85%B5%E5%9B%A2%E4%B8%80%E6%9E%AA%E4%BC%A4%E5%AE%B3%E8%AE%A1%E7%AE%97/  

