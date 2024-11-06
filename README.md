# How to Use

- install the appropriate packages
```bash
sudo apt update
sudo apt install python3
sudo apt install pip
```
- to install libraries, run
```bash
sudo pip install -r requirements.txt --break-system-packages
```
- to generate the input CSV, run this query on content DB
```sql
select distinct wlsmm.`rank`, w.id, wi.label, wd.shortForm as definition, pos.label as POS, t.`text` as japaneseTranslation, dl.transcript as canonicalLine from WordListSharedMeaningMatch wlsmm
join Word w on w.sharedMeaningId = wlsmm.sharedMeaningId and w.baseform = 1
join WordInstance wi on wi.id = w.wordInstanceId
join WordDefinition wd on wd.id = w.wordDefinitionId
join PartOfSpeech pos on pos.id = w.partOfSpeechId
join DialogLine dl on dl.id = w.canonicalDialogLineId
join DialogLineWordMatch dlwm on dlwm.WordId = w.id
left join ContentTranslationMatch ctm on ctm.contentId = w.id and ctm.contentTranslationTypeId = 41
join `Translation` t on t.id = ctm.translationId and t.languageId = 2
where wlsmm.wordListTypeId = 2 and w.active = 1 and dl.active = 1 and wd.active = 1 and wi.active = 1 and ctm.active = 1 and t.active = 1
order by wlsmm.`rank`;
```
- export the file as ngsl_check_input.csv
- run the python file via: `python ngsl.py`
- wait for a few minutes, and it will generate the file ngsl_check_output.csv

## Running on the VM
- git commit ngsl_check_input.csv into the repo then git push origin
- SSH into the vocab-checker VM
- enter these commands
```bash
sudo su root
```

- then either create a screen or attach to a current screen. to create a screen
```bash
screen -S session_name
```
- to attach to an existing screen. check the current screen list first, then attach to it
```bash
screen -ls
screen -r session_name
```
- invoke the script
```bash
cd ~/vocab-ai-checker
git pull
# openai version
python3 ngsl.py
# gemini version
python3 ngsl.py gemini
```

- when the script finishes, git commit ngsl_check_output.csv to git repo then git push origin
