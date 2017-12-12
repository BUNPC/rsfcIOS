function rsfc_excludeData_PatchCallback(Idx)

global rsfc
global obj_count

% rsfc.excludeData(idx,:) = [];
% hp = rsfc.excludeData_handle(idx);
% delete(hp);
% rsfc.excludeData_handle(idx) = [];
idx = find(obj_count==Idx);
rsfc.excludeData(idx,:) = [];
hp = rsfc.excludeData_handle(idx);
delete(hp);
rsfc.excludeData_handle(idx) = [];
obj_count(idx) = [];