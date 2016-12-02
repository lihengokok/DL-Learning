filename = 'RA162_sn.eaf';
elanInput = elanReadFile(filename);
fps = 30;
tiers = elanInput.tiers;
child_gaze = tiers.child_gaze;
unscore = tiers.eyes;
output = [];
for gaze = child_gaze;
    if ~strcmp(gaze.value,'gaze_to_eyes');
        continue;
    end
    startF = ceil(gaze.start * fps);
    endF = ceil (gaze.stop * fps);
    for f = startF:endF;
        newRow = [f 1];
        output = [output; newRow];
    end
end

for u = unscore;
    if ~strcmp(u.value,'eyes_not_visible');
        continue;
    end
    startF = ceil(u.start * fps);
    endF = ceil (u.stop * fps);
    for f = startF:endF;
        newRow = [f 2];
        output = [output; newRow];
    end
end

csvwrite('RA162_sn_un.csv', output)