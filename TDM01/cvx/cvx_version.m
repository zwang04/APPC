function varargout = cvx_version( varargin )

% CVX_VERSION   Returns version and environment information for CVX.
%
%    When called with no arguments, CVX_VERSION prints out version and
%    platform information that is needed when submitting CVX bug reports.
%
%    This function is also used internally to return useful variables that
%    allows CVX to adjust its settings to the current environment.

global cvx___

args = varargin;
compile = false;
server = false;
quick = nargout > 0;
if nargin
    if ~ischar( args{1} ),
        quick = true;
    else
        tt = strcmp( args, '-quick' );
        quick = any( tt );
        if quick, args(tt) = []; end
        tt = strcmp( args, '-compile' );
        compile = any( tt );
        if compile, quick = false; args(tt) = []; end
        tt = strcmp( args, '-server' );
        server = any( tt );
        if server, quick = false; args(tt) = []; end
    end
end

if isfield( cvx___, 'loaded' ),
    
    if quick, return; end
    fs = cvx___.fs;
    mpath = cvx___.where;
    isoctave = cvx___.isoctave;
    msub = cvx___.msub;
    
else
    
    % Matlab / Octave flag
    isoctave = exist( 'OCTAVE_VERSION', 'builtin' );

    % File and path separators, MEX extension
    if isoctave,
        comp = octave_config_info('canonical_host_type');
        mext = 'mex';
        izpc = false;
        izmac = false;
        if octave_config_info('mac'),
            msub = 'mac';
            izmac = true;
        elseif octave_config_info('windows'),
            msub = 'win';
            izpc = true;
        elseif octave_config_info('unix') && any(strfind(comp,'linux')),
            msub = 'lin';
        else
            msub = 'unknown';
        end
        if ~isempty( msub ),
            msub = [ 'o_', msub ];
            if strncmp( comp, 'x86_64', 6 ),
                msub = [ msub, '64' ];
            else
                msub = [ msub, '32' ];
            end
        end
    else
        comp = computer;
        izpc = strncmp( comp, 'PC', 2  );
        izmac = strncmp( comp, 'MAC', 3 );
        mext = mexext;
        msub = '';
    end
    if izpc,
        fs = '\'; 
        fsre = '\\';
        ps = ';'; 
        cs = false;
    else
        fs = '/'; 
        fsre = '/';
        ps = ':';
        cs = ~izmac;
    end

    % Install location
    mpath = mfilename('fullpath');
    temp = strfind( mpath, fs );
    mpath = mpath( 1 : temp(end) - 1 );

    % Numeric version
    nver = version;
    nver(nver=='.') = ' ';
    nver = sscanf(nver,'%d');
    nver = nver(1) + 0.01 * ( nver(2) + 0.01 * nver(3) );
    
    if isoctave || ~usejava('jvm'),
        jver = 0;
    else
        jver = char(java.lang.System.getProperty('java.version'));
        try
            ndxs = strfind( jver, '.' );
            jver = str2double( jver(1:ndxs(2)-1) );
        catch
            jver = 0;
        end
    end
    
    cvx___.where = mpath;
    cvx___.isoctave = isoctave;
    cvx___.nver = nver;
    cvx___.jver = jver;
    cvx___.comp = comp;
    cvx___.mext = mext;
    cvx___.msub = msub;
    cvx___.fs = fs;
    cvx___.fsre = fsre;
    cvx___.ps = ps;
    cvx___.cs = cs;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Quick exit for non-verbose output %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if quick,
    if nargout,
        varargout = { fs, cvx___.ps, mpath, cvx___.mext };
    end
    cvx_load_prefs( 0 );
    cvx___.loaded = true;
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Verbose output (cvx_setup, cvx_version plain) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cvx_ver = '3.0beta';
cvx_bld = '1151';
cvx_bdate = 'Sun Dec 7 09:28:11 2014';
cvx_bcomm = '7b8e804';
line = '---------------------------------------------------------------------------';
fprintf( '\n%s\n', line );
fprintf( 'CVX: Software for Disciplined Convex Programming       (c)2014 CVX Research\n' );
fprintf( 'Version %7s, Build %4s (%7s)%38s\n', cvx_ver, cvx_bld, cvx_bcomm, cvx_bdate );
fprintf( '%s\n', line );
fprintf( 'Installation info:\n' );
fprintf( '    Path: %s\n', cvx___.where );
if isoctave,
    fprintf( '    GNU Octave %s on %s\n', version, cvx___.comp );
else
    verd = ver('MATLAB');
    fprintf( '    MATLAB version: %s %s\n', verd.Version, verd.Release );
    if usejava( 'jvm' ),
        os_name = char(java.lang.System.getProperty('os.name'));
        os_arch = char(java.lang.System.getProperty('os.arch'));
        os_version = char(java.lang.System.getProperty('os.version'));
        java_str = char(java.lang.System.getProperty('java.version'));
        fprintf('    OS: %s %s version %s\n', os_name, os_arch, os_version );
        fprintf('    Java version: %s\n', java_str );
    else
        fprintf( '    Architecture: %s\n', cvx___.comp );
        fprintf( '    Java version: disabled\n' );
    end
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for valid version %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

issue = false;
isoctave = cvx___.isoctave;
nver = cvx___.nver;
if isoctave,
    if nver <= 3.08,
        fprintf( '%s\nCVX is not yet supported on Octave.\n(Please do not waste further time trying: changes to Octave are required.\nBut they are coming! Stay tuned.)\n%s\n', line, line );
        issue = true;
    end
elseif nver < 7.08 && strcmp( cvx___.comp(end-1:end), '64' ),
    fprintf( '%s\nCVX requires MATLAB 7.8 or later (7.5 or later on 32-bit platforms).\n' , line, line );
    issue = true;
elseif nver < 7.05,
    fprintf( '%s\nCVX requires MATLAB 7.5 or later (7.8 or later on 64-bit platforms).\n' , line, line );
    issue = true;
end

%%%%%%%%%%%%%%%%%%%%%%%%
% Verify file contents %
%%%%%%%%%%%%%%%%%%%%%%%%

fid = fopen( [ mpath, fs, 'MANIFEST' ], 'r' );
if fid > 0,
    fprintf( 'Verfying CVX directory contents:' );
    manifest = textscan( fid, '%s' );
    manifest = manifest{1};
    fclose( fid );
    newman = get_manifest( mpath, fs );
    if ~isequal( manifest, newman ),
        missing = setdiff( manifest, newman );
        additional = setdiff( newman, manifest );
        if ~isempty( missing ) || ~isempty( additional ),
            if fs ~= '/',
                missing = strrep( missing, '/', fs );
                additional = strrep( additional, '/', fs );
            end
            if ~isempty( missing ),
                fprintf( '\n    WARNING: The following files/directories are missing:\n' );
                isdir = cellfun(@(x)x(end)==fs,missing);
                missing_d = missing(isdir);
                missing_f = missing(~isdir);
                while ~isempty( missing_d ),
                    mdir = missing_d{1};
                    ss = strncmp( missing_d, mdir, length(mdir) );
                    tt = strncmp( missing_f, mdir, length(mdir) );
                    fprintf( '        %s%s%s + %d files, %d subdirectories\n', mpath, fs, mdir, nnz(tt), nnz(ss) - 1 );
                    missing_d(ss) = [];
                    missing_f(tt) = [];
                end
                for k = 1 : min(length(missing_f),10),
                    fprintf( '        %s%s%s\n', mpath, fs, missing_f{k} );
                end
                if length(missing_f) > 10,
                    fprintf( '        (and %d more files)\n', length(missing_f) - 10 );
                end
                fprintf( '    These omissions may prevent CVX from operating properly.\n'  );
            end
            if ~isempty( additional ),
                if isempty( missing ), fprintf( '\n' ); end
                fprintf( '    WARNING: The following extra files/directories were found:\n' );
                isdir = cellfun(@(x)x(end)==fs,additional);
                issedumi = cellfun(@any,regexp( additional, [ '^sedumi.*[.]', mexext, '$' ] ));
                additional_d = additional(isdir&~issedumi);
                additional_f = additional(~isdir&~issedumi);
                additional_s = additional(issedumi);
                while ~isempty( additional_d ),
                    mdir = additional_d{1};
                    ss = strncmp( additional_d, mdir, length(mdir) );
                    tt = strncmp( additional_f, mdir, length(mdir) );
                    fprintf( '        %s%s%s + %d files, %d subdirectories\n', mpath, fs, mdir, nnz(tt), nnz(ss) - 1 );
                    additional_d(ss) = [];
                    additional_f(tt) = [];
                end
                for k = 1 : min(length(additional_f),10),
                    fprintf( '        %s%s%s\n', mpath, fs, additional_f{k} );
                end
                if length(additional_f) > 10,
                    fprintf( '        (and %d more files)\n', length(additional_f) - 10 );
                end
                fprintf( '    These files may alter the behavior of CVX in unsupported ways.\n' );
                if ~isempty( additional_s ),
                    fprintf( '    ERROR: obsolete versions of SeDuMi MEX files were found:\n' );
                    for k = 1 : length(additional_s),
                        fprintf( '        %s%s%s\n', mpath, fs, additional_f{k} );
                    end
                    fprintf( '    These files are now obsolete, and must be removed to ensure\n' );
                    fprintf( '    that SeDuMi operates properly and produces sound results.\n' );
                    if ~issue,
                        fprintf( '    Please remove these files and re-run CVX_SETUP.\n' );
                        issue = true;
                    end
                end
            end
        else
            fprintf( '\n    No missing files.\n' );
        end
    else
        fprintf( '\n    No missing files.\n' );
    end
else    
    fprintf( 'Manifest missing; cannot verify file structure.\n' ) ;
end
if ~compile,
    mexpath = [ mpath, fs, 'lib', fs ];
    mext = cvx___.mext;
    if ( ~exist( [ mexpath, 'cvx_eliminate_mex.', mext ], 'file' ) || ...
         ~exist( [ mexpath, 'cvx_bcompress_mex.', mext ], 'file' ) ) && ~issue,
        issue = true;
        if ~isempty( msub ),
          mexpath = [ mexpath, msub, fs ];
          issue = ~exist( [ mexpath, 'cvx_eliminate_mex.mex' ], 'file' ) || ...
                         ~exist( [ mexpath, 'cvx_bcompress_mex.mex' ], 'file' );
        end
        if issue,
          fprintf( '    ERROR: one or more MEX files for this platform are missing.\n' );
          fprintf( '    These files end in the suffix ".%s". CVX will not operate\n', mext );
          fprintf( '    without these files. Please visit\n' );
          fprintf( '        http://cvxr.com/cvx/download\n' );
          fprintf( '    And download a distribution targeted for your platform.\n' );
        end
    end
end

%%%%%%%%%%%%%%%
% Preferences %
%%%%%%%%%%%%%%%

cvx_load_prefs( server + 1 );

%%%%%%%%%%%%%%%%
% License file %
%%%%%%%%%%%%%%%%

if isoctave,
    if ~isempty( cvx___.license ),
        fprintf( 'CVX Professional is not supported with Octave.\n' );
    end
elseif strcmpi(which('cvx_license'),[cvx___.where,fs,'cvx_license.p']),
    cvx_license( args{:} );
end

%%%%%%%%%%%%%%%
% Wrapping up %
%%%%%%%%%%%%%%%

if ~issue,
    cvx___.loaded = true;
end
clear fs;
fprintf( '%s\n', line );
if length(dbstack) <= 1,
    fprintf( '\n' );
end

%%%%%%%%%%%%%%%%%%%%%%
% Preference loading %
%%%%%%%%%%%%%%%%%%%%%%

function cvx_load_prefs( verbose )

global cvx___
fs = cvx___.fs;
isoctave = cvx___.isoctave;
errmsg = {};
if verbose,
    fprintf( 'Loading preferences:\n' );
end

pfile{1} = [ cvx___.where, fs, 'cvx_prefs.mat' ];
if isoctave,
    pfile{2} = [ prefdir, fs, '.cvx_prefs.mat' ];
else
    pfile{2} = [ regexprep( prefdir(1), [ cvx___.fsre, 'R\d\d\d\d\w$' ], '' ), fs, 'cvx_prefs.mat' ];
end
if isfield( cvx___, 'loaded' ),
    fprintf( 'Preferences already loaded:\n' );
    if ~isempty( cvx___.gprefs ),
        fprintf( '    Global: %s\n', pfile{1} );
    end
    fprintf( '    Local: %s\n', pfile{2} );
    return
end

outq = [];
first = true;
need_default = true;
for k = 1 : 2;
    gfile = pfile{k};
    if verbose,
        if k == 1, typ = 'Global'; else typ = 'Local'; end
        fprintf( '    %s: %s ...', typ, gfile );
    end
    if exist( gfile, 'file' )
        try
            outg = [];
            outg = load( gfile );
            if first,
                cvx___.path      = outg.path;
                cvx___.license   = outg.license;
                cvx___.solvers   = outg.solvers;
                cvx___.broadcast = outg.broadcast;
                cvx___.expert    = outg.expert;
                cvx___.precision = outg.precision;
                cvx___.precflag  = outg.precflag;
                cvx___.quiet     = outg.quiet;
                cvx___.profile   = outg.profile;
                need_default = false;
                if k == 1, 
                    outq = outg; 
                    outq.pfile = gfile;
                end
                first = false;
            else
                if ~isempty( outg.expert ),    cvx___.expert = outg.expert;    end
                if ~isempty( outg.precision ), cvx___.precision = outg.precision; end
                if ~isempty( outg.precflag ),  cvx___.precflag = outg.precflag;  end
                if ~isempty( outg.broadcast ), cvx___.broadcast = outg.broadcast; end
                if ~isempty( outg.quiet ),     cvx___.quiet = outg.quiet;     end
                if ~isempty( outg.profile ),   cvx___.profile = outg.profile;   end
                if ~isempty( outg.path ),      cvx___.path = outg.path;      end
                if ~isempty( outg.solvers ),   cvx___.solvers = outg.solvers;   end
                if ~isempty( outg.license ),   cvx___.license = outg.license;   end
            end
        catch exc
            if verbose,
                fprintf( ' ERROR\n' );
            else
                errmsg{end+1} = sprintf( 'Error loading %s:', gfile ); %#ok
            end
            if isempty( outg )
                errmsg = cvx_error( exc, '    ' );
            else
                errmsg{end+1} = '    File is corrupt or out-of-date.'; %#ok
                outg = [];
            end
        end
        if isempty( outg ),
            if k == 1 && verbose < 2,
                errmsg{end+1} = '    Please notify your system manager.'; %#ok
            elseif first,
                errmsg{end+1} = '    Reverting to default preferences.'; %#ok
                need_default = true;
            else
                errmsg{end+1} = '    Reverting to global preferences.'; %#ok
            end
            if verbose,
                fprintf( '    %s\n', errmsg{:} );
                errmsg = {};
            end
        elseif verbose
            fprintf( ' loaded.\n' );
        end
    elseif k == 3 - verbose,
        fprintf( ' to be created.\n' );
    elseif verbose
        fprintf( ' not found.\n' );
    end
end

if ~isempty( errmsg )
    temp = sprintf( '%s\n', errmsg{:} );
    cvx_throw( temp(1:end-1) );
end

if need_default,
    cvx___.expert    = false;
    cvx___.precision = [eps^0.5,eps^0.5,eps^0.25];
    cvx___.precflag  = 'default';
    cvx___.broadcast = isoctave;
    cvx___.quiet     = false;
    cvx___.profile   = false;
    cvx___.path      = [];
    cvx___.solvers   = [];
    cvx___.license   = [];
end

cvx___.pfile = pfile{end};
cvx___.gprefs = outq;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recursive manifest building function %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function newman = get_manifest( mpath, fs )
dirs   = {};
files  = {};
nfiles = dir( mpath );
ndir   = '';
dndx   = 0;
pat2   = '^\.|~$|';
pat    = '^\.|~$|^cvx_license.[md]at$|^doc$|^examples$';
while true,
    isdir  = [ nfiles.isdir ];
    nfiles = { nfiles.name };
    tt     = cellfun( @isempty, regexp( nfiles, pat ) ); pat = pat2;
    isdir  = isdir(tt);
    nfiles = nfiles(tt);
    ndirs  = nfiles(isdir);
    if ~isempty(ndirs),
        dirs = horzcat( dirs, strcat(strcat(ndir,ndirs), fs ) ); %#ok
    end
    nfiles = nfiles(~isdir);
    if ~isempty(nfiles),
        files = horzcat( files, strcat(ndir,nfiles) ); %#ok
    end
    if length( dirs ) == dndx, break; end
    dndx = dndx + 1;
    ndir = dirs{dndx};
    nfiles = dir( [ mpath, fs, ndir ] );
end
[tmp,ndxs1] = sort(upper(dirs)); %#ok
[tmp,ndxs2] = sort(upper(files)); %#ok
newman = horzcat( dirs(ndxs1), files(ndxs2) );
if fs ~= '/',
    newman = strrep( newman, fs, '/' );
end
newman = newman(:);

% Copyright 2005-2014 CVX Research, Inc.
% See the file LICENSE.txt for full copyright information.
% The command 'cvx_where' will show where this file is located.
