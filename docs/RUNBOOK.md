# Runbook

Как поставить, запустить и проверить. Команды кроссплатформенные, Makefile
намеренно нет.

## Требования

- Python 3.11 или 3.12. Верхняя граница — требование `flyvis` (`<3.13`).
- [uv](https://docs.astral.sh/uv/).
- Для полной скорости — ускоритель (CUDA или Apple MPS). На CPU всё работает, но
  медленнее, см. `docs/M0_REPORT.md`.

## Установка

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Зрительный путь ставится отдельно, он тяжёлый:

```bash
uv pip install --python .venv/bin/python -e ".[flyvis]"
```

Проверка, что всё встало:

```bash
.venv/bin/flymog probe-env
.venv/bin/pytest -q
```

## Чек-лист M0 на твоей машине

Порядок важен: каждая команда пишет JSON в `artifacts/m0/`, из которого потом
собирается отчёт.

```bash
# 1. Железо. Покажет, есть ли CUDA или MPS и какой backend будет выбран.
.venv/bin/flymog probe-env

# 2. Что из больших данных на месте и какие хосты доступны.
#    Напечатает точные ссылки и пути, если чего-то не хватает.
.venv/bin/flymog fetch-data

# 3. Какие входные клетки есть в аннотациях. Нужен только файл аннотаций.
.venv/bin/flymog probe-inputs

# 4. Сшивается ли flyvis с FlyWire. Главный вопрос M0.
.venv/bin/flymog probe-bridge

# 5. Скорость. Долго. На суррогате, если настоящих связей ещё нет.
.venv/bin/flymog bench-sim --surrogate --batch 32 --check-pruning

# 6. Режим работы сети. Батч бери большим, иначе критерий вырождается.
.venv/bin/flymog sweep-regime --surrogate --batch 128

# 7. Собрать отчёт из всего, что получилось.
.venv/bin/flymog m0-report
```

Отчёт окажется в `docs/M0_REPORT.md`. Разделы, для которых нет JSON, будут
помечены как незаполненные — это нормально и означает ровно то, что написано.

## Загрузка данных

Zenodo, Codex FlyWire, HuggingFace и Google Drive могут быть закрыты политикой
сети. Но всё нужное, кроме весов flyvis, лежит на GitHub и берётся обычным git.

```bash
# Связи и порядок нейронов (FlyWire v783, ~370 МБ вместе с историей)
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git
cp Drosophila_brain_model/Connectivity_783.parquet data/connectome/
cp Drosophila_brain_model/Completeness_783.csv     data/connectome/

# Аннотации: типы клеток, суперклассы, координаты
git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git
cp flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
   data/connectome/flywire_neurons.tsv
```

Проверить, что всё на месте:

```bash
.venv/bin/flymog fetch-data
```

| Файл | Куда | Что даёт |
|---|---|---|
| `Connectivity_783.parquet` | `data/connectome/` | 15 091 983 связи со знаком; после порога ≥5 остаётся 2 700 513 |
| `Completeness_783.csv` | `data/connectome/` | 138 639 нейронов, задаёт порядок индексов |
| `flywire_neurons.tsv` | `data/connectome/` | типы клеток, суперклассы, координаты |
| Веса flyvis | каталог результатов flyvis | **TODO(verify)**, хост был закрыт |

Аннотаций одних достаточно для `probe-inputs` и `probe-bridge`; для `bench-sim`
и `sweep-regime` нужны связи.

Перед использованием прочитай `docs/DATA_LICENSES.md`: лицензия данных FlyWire
на момент M0 **не установлена окончательно**.

## Быстрый взгляд на зрение мухи

```bash
.venv/bin/flymog hex-demo path/to/photo.jpg --output artifacts/hex_demo.png
```

Покажет, как кадр выглядит после гекса-решётки из 721 колонки. Полезно, чтобы
глазами проверить кадрирование и долю поля зрения.

## Бот

Ещё не реализован, появится в M3. Тогда здесь будут запуск, остановка, логи и
автозапуск (systemd / launchd / Task Scheduler).

Токен бота кладётся **только** в `.env` (см. `.env.example`). `.env` в
`.gitignore`. Никогда не отправляй токен в чат и не коммить его.

## Если что-то сломалось

| Симптом | Что смотреть |
|---|---|
| `flymog` не найден | окружение не активировано, зови `.venv/bin/flymog` |
| probe падает с `ConnectomeDataMissing` | сообщение содержит точные ссылки и пути, это не ошибка кода |
| `probe-bridge` не запускается | не установлен `flyvis`, поставь extra или передай `--flyvis-types-file` |
| Сеть молчит, частоты нулевые | тяга ниже порога, гони `sweep-regime` |
| Все PR одинаковы | батч слишком мал, probe предупредит об этом сам |
| Скорость сильно ниже цели | смотри лестницу ускорения в `docs/M0_REPORT.md` |
