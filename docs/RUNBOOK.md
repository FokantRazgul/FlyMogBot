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

## Ручная загрузка данных

`flymog fetch-data` печатает, что именно нужно и куда положить. Кратко:

| Файл | Куда | Откуда |
|---|---|---|
| Аннотации нейронов | `data/connectome/flywire_neurons.tsv` | репозиторий `flyconnectome/flywire_annotations`, файл `supplemental_files/Supplemental_file1_neuron_annotations.tsv` |
| Таблица связей | `data/connectome/flywire_connections.csv` | Zenodo, запись FlyWire. **TODO(verify)** точный файл |
| Веса flyvis | каталог результатов flyvis | см. документацию flyvis. **TODO(verify)** |

Аннотации можно получить обычным git:

```bash
git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git
cp flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
   data/connectome/flywire_neurons.tsv
```

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
