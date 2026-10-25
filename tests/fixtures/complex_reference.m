% Generate golden values from the original, commented MATLAB formulas.
% Run from the project root:
% octave-cli --no-init-file --quiet --no-history tests/fixtures/complex_reference.m
source = fileread('mLF/+mlf/examples.m');
entries = regexp(source, '(?m)^    % case (\d+)\s*\n    %     H\s*=\s*([^\n]+);', 'tokens');
fprintf('[\n');
for k=1:numel(entries)
    id = str2double(entries{k}{1});
    text = strtrim(entries{k}{2});
    H = str2func(text);
    params = regexp(text, '^@\(([^)]+)\)', 'tokens');
    n = numel(strsplit(params{1}{1}, ','));
    fprintf('  {"id":%d,"values":[',id);
    for sample=0:2
        j=1:n;
        x=(0.75+0.025*j+0.11*sample)+1i*(0.05+0.007*j-0.02*sample);
        args=num2cell(x);
        y=H(args{:});
        if sample>0, fprintf(','); end
        fprintf('[%.17g,%.17g]',real(y),imag(y));
    end
    fprintf(']}');
    if k<numel(entries), fprintf(','); end
    fprintf('\n');
end
fprintf(']\n');
